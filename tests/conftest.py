
import os
from langchain_deepseek import ChatDeepSeek
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import pytest
from dotenv import load_dotenv

load_dotenv()

def get_openai():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL_NAME"),
    )
def get_azure_openai():
    
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_4O_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_4O_MODEL", None),
        max_retries=5,
        temperature=0.0,
        max_tokens=os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096),
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
def get_deepseek():
    return ChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        temperature=0.0,
        max_completion_tokens=10000,
        max_retries=3,
    )

@pytest.fixture(scope="module")
def llm():    
    return get_openai()