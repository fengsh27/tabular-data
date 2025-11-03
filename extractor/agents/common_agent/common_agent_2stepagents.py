import os
from typing import Any, Callable, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_incrementing
import logging

from extractor.agents.agent_utils import escape_braces_for_format
from extractor.agents.common_agent.common_agent_2steps import CommonAgentTwoSteps
from extractor.llm import structured_output_llm
from .common_agent import (
    CommonAgent,
    RetryException,
)
from extractor.constants import COT_USER_INSTRUCTION

logger = logging.getLogger(__name__)

def get_openai():
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_MODEL", None),
        max_retries=5,
        temperature=0.0,
        max_completion_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096)),
        top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
    )
    


class CommonAgentTwoChainStepAgents(CommonAgent):
    def __init__(self, llm: BaseChatOpenAI):
        super().__init__(llm)
        # self.azure_llm = get_openai()

    def _initialize(self):
        self.exceptions = None
        self.token_usage = None

    def _get_retryexception_message(self) -> list[tuple[str, str]]:
        if self.exceptions is None:
            return None
        return [("human", str(excp)) for excp in self.exceptions]

    def _build_prompt_for_cot_step(
        self,
        system_prompt: str,
        instruction_prompt: str,
    ):
        system_prompt = escape_braces_for_format(system_prompt)
        instruction_prompt = escape_braces_for_format(instruction_prompt)
        msgs = [("system", system_prompt)]
        msgs = msgs + [("human", instruction_prompt)]
        exception_msgs = self._get_retryexception_message()
        if exception_msgs is not None:
            msgs = msgs + exception_msgs
        msgs = msgs + [("human", COT_USER_INSTRUCTION)]
        return ChatPromptTemplate.from_messages(msgs)
    
    def _build_prompt_for_final_step(
        self,
        system_prompt: str,
        cot_msg: str,
    ):
        system_prompt = escape_braces_for_format(system_prompt)
        msgs = [("system", system_prompt)]
        cot_msg = escape_braces_for_format(cot_msg)
        msgs = msgs + [(
            "human",
            f"Please review the following step-by-step reasoning and provide the answer based on it: ```{cot_msg}```"
        )]
        return ChatPromptTemplate.from_messages(msgs)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_incrementing(start=1.0, increment=3, max=10),
    )
    def _invoke_agent(
        self,
        system_prompt: str,
        instruction_prompt: str,
        schema: any,
        post_process: Optional[Callable] = None,
        **kwargs: Optional[Any],
    ):
        # Initialize the callback handler
        callback_handler = OpenAICallbackHandler()
        cot_prompt = self._build_prompt_for_cot_step(
            system_prompt=system_prompt, 
            instruction_prompt=instruction_prompt
        )

        try:
            # First, use llm to do CoT
            msgs = cot_prompt.invoke(input={}).to_messages()
            
            # cot_res = self.llm.generate(messages=[msgs])
            cot_res = self.llm.invoke(msgs)
            reasoning_process = cot_res.content # cot_res.generations[0][0].text
            token_usage = cot_res.usage_metadata # cot_res.llm_output.get("token_usage")
            input_tokens = token_usage.get("input_tokens", 0)
            output_tokens = token_usage.get("output_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            cot_tokens = {
                "total_tokens": total_tokens,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }
            self._incre_token_usage(cot_tokens)
        except Exception as e:
            logger.error(str(e))
            raise e
        
        # Then use the reasoning process to do the structured output
        updated_prompt = self._build_prompt_for_final_step(
            system_prompt=system_prompt,
            cot_msg=reasoning_process,
        )
        # agent = updated_prompt | self.azure_llm.with_structured_output(schema)
        agent = structured_output_llm(self.llm, schema, updated_prompt)
        try:
            res = agent.invoke(
                input={},
                config={
                    "callbacks": [callback_handler],
                },
            )
            self._incre_token_usage(callback_handler)
        except Exception as e:
            logger.error(str(e))
            raise e
        processed_res = None
        if post_process is not None:
            try:
                processed_res = post_process(res, **kwargs)
            except RetryException as e:
                logger.error(str(e))
                self.exceptions = [e] if self.exceptions is None else self.exceptions + [e]
                raise e
            except Exception as e:
                logger.error(str(e))
                raise e
        return res, processed_res, self.token_usage, reasoning_process
    


