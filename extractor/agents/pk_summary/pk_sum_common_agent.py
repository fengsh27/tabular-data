
import os
from typing import Any, Callable, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_meta import (
    ChatMetaLlama, 
    meta_agent_factory,
)
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_incrementing
import logging

from extractor.utils import escape_braces_for_format
from extractor.agents.common_agent.common_agent import RetryException, CommonAgent

logger = logging.getLogger()

def get_azure_openai():
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

class PKSumCommonAgentResult(BaseModel):
    reasoning_process: str = Field(
        description="A detailed explanation of the thought process or reasoning steps taken to reach a conclusion."
    )


class PKSumCommonAgent(CommonAgent):
    def __init__(
        self, 
        llm: BaseChatOpenAI | ChatMetaLlama | ChatGoogleGenerativeAI | ChatAnthropic
    ):
        super().__init__(llm)
        if isinstance(llm, ChatMetaLlama):
            self.structured_llm = get_azure_openai()

    def _invoke_structured_llm(
        self, 
        system_prompt: str,
        instruction_prompt: str,
        schema: any,
    ):
        assert schema is not None, "schema is required"
        assert system_prompt is not None, "system_prompt is required"
        assert instruction_prompt is not None, "instruction_prompt is required"

        if not isinstance(self.llm, ChatMetaLlama):
            res, token_usage, reasoning_process = self._invoke_structured_llm_for_openai(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
            )
        else:
            res, token_usage, reasoning_process = self._invoke_structured_llm_for_meta_llama(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
            )
        if reasoning_process is None:
            if hasattr(res, "reasoning_process"):
                reasoning_process = res.reasoning_process
            else:
                reasoning_process = ""
        return res, token_usage, reasoning_process

    def _invoke_structured_llm_for_openai(
        self, 
        system_prompt: str,
        instruction_prompt: str,
        schema: any,
    ):
        system_prompt = escape_braces_for_format(system_prompt)
        instruction_prompt = escape_braces_for_format(instruction_prompt)
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", instruction_prompt),
            ])
            updated_prompt = self._process_retryexception_message(prompt)
            agent = updated_prompt | self.llm.with_structured_output(schema)
            callback_handler = OpenAICallbackHandler()
            res = agent.invoke(
                input={},
                config={
                    "callbacks": [callback_handler],
                },
            )   
            self._incre_token_usage(callback_handler)
            return res, self.token_usage, None
        except Exception as e:
            logger.error(str(e))
            raise e

    def _invoke_structured_llm_for_meta_llama(
        self, 
        system_prompt: str,
        instruction_prompt: str,
        schema: any,
    ):
        assert isinstance(self.llm, ChatMetaLlama), "llm must be a ChatMetaLlama instance"
        system_prompt = escape_braces_for_format(system_prompt)
        instruction_prompt = escape_braces_for_format(instruction_prompt)
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", instruction_prompt),
            ])
            updated_prompt = self._process_retryexception_message(prompt)

            # First, use the meta llama to do CoT
            response = self.llm.invoke(updated_prompt.format_messages())
            cot_msg = response.content
            token_usage = response.usage_metadata
            input_tokens = token_usage.get("input_tokens", 0)
            output_tokens = token_usage.get("output_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            cot_tokens = {
                "total_tokens": total_tokens,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }
            self._incre_token_usage(cot_tokens)
            cot_msg = escape_braces_for_format(cot_msg)
        except Exception as e:
            logger.error(str(e))
            raise e

        # Then, use the structured llm to do the final answer
        msgs = [(
            "system",
            system_prompt,
        ), (
            "human",
            f"Please review the following step-by-step reasoning and provide the answer based on it: ```{cot_msg}```"
        )]
        final_prompt = ChatPromptTemplate.from_messages(msgs)
        agent = final_prompt | self.structured_llm.with_structured_output(schema)
        try:
            callback_handler = OpenAICallbackHandler()
            res = agent.invoke(
                input={},
                config={
                    "callbacks": [callback_handler],
                },
            )
            self._incre_token_usage(callback_handler)
            return res, self.token_usage, cot_msg
        except Exception as e:
            logger.error(str(e))
            raise e

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
        res, token_usage, reasoning_process = self._invoke_structured_llm(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            schema=schema,
        )
        
        processed_res = None
        if post_process is not None:
            try:
                processed_res = post_process(res, **kwargs)
            except RetryException as e:
                logger.error(str(e))
                self.exception = e
                raise e
            except Exception as e:
                logger.error(str(e))
                raise e
        return res, processed_res, self.token_usage, reasoning_process
