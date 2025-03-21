
from typing import Any, Callable, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
from tenacity import (
    retry, stop_after_attempt, wait_incrementing
)

from extractor.agents.agent_utils import display_md_table

class RetryException(Exception):
    """ Exception need to retry """
    pass

class PKSumCommonAgentResult(BaseModel):
    reasoning_process: str = Field(description="A detailed explanation of the thought process or reasoning steps taken to reach a conclusion.")

class PKSumCommonAgent:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm
        self.exception: RetryException | None = None

    def go(
        self, 
        system_prompt: str, 
        instruction_prompt: str, 
        schema: any,
        pre_process: Optional[Callable]=None,
        post_process: Optional[Callable]=None,
        **kwargs: Optional[Any],
    ):
        """
        execute agent

        Args:
        system_prompt str: system prompt
        instruction_prompt str: user prompt to guide how llm execute agent

        Return:
        (drug_combination list[list[str]], reasoning_thoughts str)
        """
        self.exception = None
        if pre_process is not None:
            is_OK = pre_process(**kwargs)
            if not is_OK: # skip
                return
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", instruction_prompt),
        ])
                
        res = self._invoke_agent(
            prompt,
            schema,
            post_process,
            **kwargs,
        )
        return res
    
    def _process_retryexception_message(self, prompt: ChatPromptTemplate) -> ChatPromptTemplate:
        if self.exception is None:
            return prompt
        
        existing_messages = prompt.messages
        updated_messages = existing_messages + [("user", str(self.exception))]
        updated_prompt = ChatPromptTemplate.from_messages(updated_messages)
        return updated_prompt
    
    @retry(stop=stop_after_attempt(3), wait=wait_incrementing())
    def _invoke_agent(
        self,
        prompt: ChatPromptTemplate,
        schema: any,
        post_process: Optional[Callable]=None,
        **kwargs: Optional[Any],
    ):
        updated_prompt = self._process_retryexception_message(prompt)
        agent = updated_prompt | self.llm.with_structured_output(schema)
        res = agent.invoke(input={})
        processed_res = None
        if post_process is not None:
            try:
                processed_res = post_process(res, **kwargs)
            except RetryException as e:
                self.exception = e
                raise e
        return res, processed_res
    

    

