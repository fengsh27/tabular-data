
from typing import Any, Callable, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_incrementing
import logging

from extractor.agents.agent_utils import increase_token_usage
from extractor.llm_utils import structured_output_llm
from extractor.utils import escape_braces_for_format

logger = logging.getLogger(__name__)

class RetryException(Exception):
    """Exception need to retry"""

    pass

class CommonAgentResult(BaseModel):
    reasoning_process: str = Field(
        description="A detailed explanation of the thought process or reasoning steps taken to reach a conclusion."
    )

class CommonAgent:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm
        self.exceptions: list[RetryException] | None = None
        self.token_usage: dict | None = None
        self.try_fix_error: Optional[Callable[[Any], Any]] = None

    def go(
        self,
        system_prompt: str,
        instruction_prompt: str,
        schema: any,
        schema_basemodel: Optional[BaseModel] = None,
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
        
        return self._invoke_agent(
            system_prompt,
            instruction_prompt,
            schema,
            schema_basemodel,
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
        incremental_token_usage = token_usage
        if not isinstance(token_usage, dict):
            incremental_token_usage = vars(incremental_token_usage)
        self.token_usage = increase_token_usage(
            self.token_usage, incremental_token_usage
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_incrementing(start=1.0, increment=3, max=10),
    )
    def _invoke_agent(
        self,
        system_prompt: str,
        instruction_prompt: str,
        schema: any,
        schema_basemodel: Optional[BaseModel] = None,
        post_process: Optional[Callable] = None,
        **kwargs: Optional[Any],
    ) -> tuple[Any, Any, dict | None, Any | None]:
        system_prompt = escape_braces_for_format(system_prompt)
        instruction_prompt = escape_braces_for_format(instruction_prompt)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", instruction_prompt),
        ])
        # Initialize the callback handler
        callback_handler = OpenAICallbackHandler()

        updated_prompt = self._process_retryexception_message(prompt)
        agent = structured_output_llm(self.llm, schema, updated_prompt)
        # agent = updated_prompt | self.llm.with_structured_output(schema)
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
        processed_res = res
        if post_process is not None:
            try:
                processed_res = post_process(res, **kwargs)
            except RetryException as e:
                logger.error(str(e))
                if self.try_fix_error is not None and self.exceptions is not None and len(self.exceptions) == 4:
                    fixed_res = self.try_fix_error(res, **kwargs)
                    if fixed_res is not None:
                        return res, fixed_res, self.token_usage, None
                if self.exceptions is None:
                    self.exceptions = [e]
                else:
                    self.exceptions.append(e)
                raise e
            except Exception as e:
                logger.error(str(e))
                raise e
        return res, processed_res, self.token_usage, None
    