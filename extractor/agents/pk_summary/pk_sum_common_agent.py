from typing import Any, Callable, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_incrementing
import logging

from extractor.agents.agent_utils import (
    increase_token_usage,
)

logger = logging.getLogger()


class RetryException(Exception):
    """Exception need to retry"""

    pass


class PKSumCommonAgentResult(BaseModel):
    # reasoning_process: list[str] = Field(
    #     description="A list of strings for detailed explanation of the thought process or reasoning steps taken to reach a conclusion."
    # )
    pass


class PKSumCommonAgent:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm
        self.exceptions: list[RetryException] | None = None
        self.token_usage: dict | None = None

    def go(
        self,
        system_prompt: str,
        instruction_prompt: str,
        schema: any,
        pre_process: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        **kwargs: Optional[Any],
    ):
        """
        execute agent

        Args:
        system_prompt str: system prompt
        instruction_prompt str: user prompt to guide how llm execute agent
        schema pydantic.BaseModel or json schema: llm output result schema
        pre_process Callable or None: pre-processor that would be executed before llm.invoke
        post_process Callable or None: post-processor that would be executed after llm.invoke
        kwargs None or dict: args for pre_proces and post_process

        Return:
        (output that comply with input args `schema`)
        """
        self._initialize()
        if pre_process is not None:
            is_OK = pre_process(**kwargs)
            if not is_OK:  # skip
                return
        
        return self._invoke_agent(
            system_prompt,
            instruction_prompt,
            schema,
            post_process,
            **kwargs,
        )

    def _initialize(self):
        self.exceptions = None
        self.token_usage = None
    
    def _get_retryexception_message(self) -> list[tuple[str, str]]:
        if self.exceptions is None:
            return None
        return [("human", str(excp)) for excp in self.exceptions]
    
    def _append_COT_message(
        self, prompt: ChatPromptTemplate, cot_msg: str,
    ):
        if cot_msg is None or len(cot_msg.strip()) == 0:
            return prompt
        cot_msg =  cot_msg.replace("{", "{{").replace("}", "}}")
        updated_messages = prompt.messages + [(
            "human", 
            f"Please review the following step-by-step reasoning and provide the final answer based on it: \n{cot_msg}"
        )]
        return ChatPromptTemplate.from_messages(updated_messages)
    

    def _incre_token_usage(self, token_usage: OpenAICallbackHandler | dict):
        if isinstance(token_usage, OpenAICallbackHandler):
            usage = {
                "total_tokens": token_usage.total_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "prompt_tokens": token_usage.prompt_tokens,
            }
        else:
            usage = token_usage
        self.token_usage = increase_token_usage(self.token_usage, usage)

    def _build_prompt(
        self, 
        system_prompt: str, 
        instruction_prompt: str | None = None,
        cot_msg: str | None = None,
    ):
        msgs = [
            ("system", system_prompt),
        ]
        msgs = msgs if instruction_prompt is None else msgs + [("human", instruction_prompt)]
        msgs = msgs if cot_msg is None else msgs + [(
            "human", 
            f"Please review the following step-by-step reasoning and provide the final answer based on it: \n{cot_msg.replace('{', '{{').replace('}', '}}')}"
        )]
        if instruction_prompt is not None:
            exception_msgs = self._get_retryexception_message()
            msgs = msgs if exception_msgs is None else msgs + exception_msgs

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
        cot_prompt = self._build_prompt(
            system_prompt=system_prompt, 
            instruction_prompt=instruction_prompt
        )

        msgs = cot_prompt.invoke(input={}).to_messages()
        # First, use llm to do CoT
        try:
            cot_res = self.llm.generate(messages=[msgs])
            reasoning_process = cot_res.generations[0][0].text
            token_usage = cot_res.llm_output.get("token_usage")
            cot_tokens = {
                "total_tokens": token_usage.get("total_tokens", 0),
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
            }
            self._incre_token_usage(cot_tokens)
        except Exception as e:
            logger.error(str(e))
            raise e
        
        updated_prompt = self._build_prompt(
            system_prompt=system_prompt,
            cot_msg=reasoning_process,
        )
        agent = updated_prompt | self.llm.with_structured_output(schema)
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
