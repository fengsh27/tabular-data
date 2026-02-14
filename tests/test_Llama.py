import pytest
import os
from langchain_meta import (
    ChatMetaLlama, 
    meta_agent_factory,
)
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LlamaTestResult(BaseModel):
    result: str = Field(description="The answer to the question")
    reasoning_process: str = Field(description="The reasoning process of the answer")

@pytest.mark.skip()
def test_MetaLlama():
    api_key = os.environ.get("LLAMA_API_KEY")
    model = os.environ.get("LLAMA_MODEL")
    llm = ChatMetaLlama(
        model=model,
        api_key=api_key,
        base_url="https://api.llama.com/v1/",
        temperature=0,
        max_tokens=64000,
    )

    response = llm.invoke([HumanMessage(content="Hello Llama!")])
    logger.info(response.content)

@pytest.mark.skip()
def test_MetaLlama_structured_output():
    api_key = os.environ.get("LLAMA_API_KEY")
    model = os.environ.get("LLAMA_MODEL")
    llm = ChatMetaLlama(
        model=model,
        api_key=api_key,
        base_url="https://api.llama.com/v1/",
        temperature=0,
        max_tokens=64000,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", "What is the capital of France? Before arriving at the answer, please think step by step."),
        ]
    )
    # chain = prompt | llm.with_structured_output(LlamaTestResult)
    try:
        # res = chain.invoke({})
        res = llm.invoke(prompt.format_messages())
    except Exception as e:
        logger.error(e)
    else:
        logger.info(res.content)
        logger.info(res.usage_metadata)

@pytest.mark.skip()
def test_MetaLlama_isinstance():
    api_key = os.environ.get("LLAMA_API_KEY")
    model = os.environ.get("LLAMA_MODEL")
    llm = ChatMetaLlama(
        model=model,
        api_key=api_key,
        base_url="https://api.llama.com/v1/",
        temperature=0,
        max_tokens=64000,
    )
    llm_gemini = ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("GEMINI_MODEL"),
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    is_base_chat_openai = isinstance(llm, BaseChatOpenAI)
    logger.info(is_base_chat_openai)
    assert not is_base_chat_openai

    is_base_chat_openai = isinstance(llm_gemini, BaseChatOpenAI)
    logger.info(is_base_chat_openai)
    assert is_base_chat_openai

# Example with structured output
class ResponseSchema(BaseModel):
    answer: str
    confidence: float

@pytest.mark.skip(reason="This test relies on LLM API key")
def test_MetaLlama_agent_factory(azure_llm):
    api_key = os.environ.get("LLAMA_API_KEY")
    model = os.environ.get("LLAMA_MODEL")
    llm = ChatMetaLlama(
        model=model,
        api_key=api_key,
        base_url="https://api.llama.com/v1/",
        temperature=0,
        max_tokens=64000,
    )

    system_prompt_text = """
You are a helpful assistant.

Please always respond in the following JSON format:
{{
  "answer": "...",
  "confidence": <float between 0 and 1>
}}
"""
    structured_agent = meta_agent_factory(
        llm=llm,
        output_schema=ResponseSchema,
        system_prompt_text=system_prompt_text,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            ("human", "What is the capital of France? Before arriving at the answer, please think step by step."),
        ]
    )
    # structured_agent.
    response = llm.invoke(prompt.format_messages())
    reasoning_process = response.content.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            ("human", "What is the capital of France? Before arriving at the answer, please think step by step."),
            ("human", f"Please review the following reasoning process and provide the answer based on it: \n {reasoning_process}" ),
        ]
    )

    prompt_with_structured_output = prompt | azure_llm.with_structured_output(ResponseSchema)
    final_response = prompt_with_structured_output.invoke(
        input={},
    )

    logger.info(final_response)
    




