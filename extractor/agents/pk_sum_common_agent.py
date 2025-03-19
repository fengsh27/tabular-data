
from typing import Any, Callable, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
from tenacity import (
    retry, stop_after_attempt, wait_incrementing
)

from extractor.agents.agent_utils import display_md_table

class PKSumCommonAgentResult(BaseModel):
    reasoning_process: str = Field(description="A detailed explanation of the thought process or reasoning steps taken to reach a conclusion.")

@retry(stop=stop_after_attempt(3), wait=wait_incrementing())
def _invoke_agent(
    llm: BaseChatOpenAI,
    prompt: ChatPromptTemplate,
    schema: any,
    post_process: Optional[Callable]=None,
    **kwargs: Optional[Any],
):
    agent = prompt | llm.with_structured_output(schema)
    res = agent.invoke(input={})
    if post_process is not None:
        res = post_process(res, **kwargs)
    return res

class PKSumCommonAgent:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def go(
        self, 
        system_prompt: str, 
        instruction_prompt: str, 
        schema: any,
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
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", instruction_prompt),
        ])
                
        res = _invoke_agent(
            self.llm,
            prompt,
            schema,
            post_process,
            **kwargs,
        )
        return res
    

    

