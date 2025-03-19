
from typing import Any, Callable, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from extractor.agents.agent_utils import display_md_table

class PKSumCommonAgentResult(BaseModel):
    reasoning_process: str = Field(description="A detailed explanation of the thought process or reasoning steps taken to reach a conclusion.")

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
        agent = prompt | self.llm.with_structured_output(schema)
        res = agent.invoke(input={})
        if post_process is not None:
            res = post_process(res, **kwargs)
        return res
    

    

