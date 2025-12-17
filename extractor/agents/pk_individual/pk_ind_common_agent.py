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
from extractor.agents.common_agent.common_agent import RetryException

logger = logging.getLogger(__name__)




class PKIndCommonAgentResult(BaseModel):
    reasoning_process: str = Field(
        description="A concise explanation (**not more than 100 tokens**) of the thought process or reasoning steps taken to reach a conclusion."
    )


class PKIndCommonAgent:
    def __init__(self, llm: BaseChatOpenAI, llm2: BaseChatOpenAI = None):
        self.llm = llm
        self.llm2 = llm2
        self.exceptions: list[RetryException] | None = None
        self.token_usage: dict | None = None
        self.try_fix_error: Optional[Callable[[Any], Any]] = None
        
    def go(
        self,
        system_prompt: str,
        instruction_prompt: str,
        schema: any,
        try_fix_error: Optional[Callable[[Any], Any]] = None,
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
        self.try_fix_error = try_fix_error
        if pre_process is not None:
            is_OK = pre_process(**kwargs)
            if not is_OK:  # skip
                return
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", instruction_prompt),
            ]
        )

        return self._invoke_agent(
            prompt,
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

    def _process_retryexception_message(
        self, prompt: ChatPromptTemplate
    ) -> ChatPromptTemplate:
        if self.exceptions is None:
            return prompt

        exception_msgs = self._get_retryexception_message()
        if exception_msgs is not None:
            existing_messages = prompt.messages
            updated_messages = existing_messages + exception_msgs
            updated_prompt = ChatPromptTemplate.from_messages(updated_messages)
            return updated_prompt
        return prompt

    def _incre_token_usage(self, token_usage):
        self.token_usage = increase_token_usage(
            self.token_usage,
            {
                "total_tokens": token_usage.total_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "prompt_tokens": token_usage.prompt_tokens,
            },
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_incrementing(start=1.0, increment=3, max=10),
    )
    def _invoke_agent(
        self,
        prompt: ChatPromptTemplate,
        schema: any,
        post_process: Optional[Callable] = None,
        **kwargs: Optional[Any],
    ):
        # Initialize the callback handler
        callback_handler = OpenAICallbackHandler()

        updated_prompt = self._process_retryexception_message(prompt)
        if self.llm2 is not None and self.exceptions is not None and len(self.exceptions) > 2:
            agent = updated_prompt | self.llm2.with_structured_output(schema)
        else:
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
                if self.exceptions is not None and len(self.exceptions) == 4 and self.try_fix_error is not None:
                    fixed_res = self.try_fix_error(res, **kwargs)
                    if fixed_res is not None:
                        return res, fixed_res, self.token_usage
                logger.error(str(e))
                if self.exceptions is None:
                    self.exceptions = [e]
                else:
                    self.exceptions.append(e)
                raise e
            except Exception as e:
                logger.error(str(e))
                raise e
        return res, processed_res, self.token_usage
