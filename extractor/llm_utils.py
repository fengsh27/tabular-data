from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama

from extractor.request_gpt_oss import get_gpt_oss
from extractor.request_openai import get_openai, get_5_openai

def get_gpt_oss_llm():
    return get_gpt_oss()

def get_pipeline_llm():
    return get_openai() # get_gpt_oss_llm() # 

def get_agent_llm():
    return get_5_openai() # get_gpt_oss_llm() # 

def structured_output_llm(
    llm,
    schema: Any,
    prompt: ChatPromptTemplate,
):
    if isinstance(llm, BaseChatOpenAI):
        agent = prompt | llm.with_structured_output(schema)
        return agent
    elif isinstance(llm, ChatOllama):
        parser = PydanticOutputParser(pydantic_object=schema)
        agent = prompt | llm | parser
        return agent
    else:
        raise ValueError(f"Unsupported LLM: {llm}")

