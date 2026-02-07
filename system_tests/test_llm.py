from typing import List, Any, Optional
# from openai import AzureOpenAI, OpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
import pytest

load_dotenv()

@pytest.mark.skip()
def test_openai():
    llm = AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_4O_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_4O_MODEL", None),
        max_retries=5,
        temperature=0.0,
        max_tokens=16384,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
    res = llm.generate(
        messages=[ [{"role": "user", "content": "Hi"}] ],
    )
    print(res)

@pytest.mark.skip()
def test_openai_1():
    api_key = os.environ.get("OPENAI_API_KEY")
    endpoint = "https://pharmacoinfo-openai-2.openai.azure.com/"
    deployment = "gpt-4o"
    model="gpt-4o"
    api_version="2024-10-21"
    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment,
        model=model,
        max_retries=5,
        temperature=0.0,
        max_tokens=16384,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
    res = llm.generate(
        messages=[ [{"role": "user", "content": "Hi"}] ],
    )
    print(res)

# @pytest.mark.skip()
def test_gemini_list_models():
    from langchain_google_genai import ChatGoogleGenerativeAI

    if "GEMINI_API_KEY" in os.environ:        
        client = ChatGoogleGenerativeAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("GEMINI_MODEL"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        messages = [
            HumanMessage("Hi")
        ]
        res = client.generate(
            messages=[ messages ],
        )
        print(res)

@pytest.mark.skip()
def test_llm():
    from typing import List, Any
    from openai import AzureOpenAI, OpenAI
    import os
    import logging
    from extractor.utils import concate_llm_contents
    logger = logging.getLogger(__name__)
    openai_type = os.environ.get("OPENAI_API_TYPE")
    if openai_type == "azure":
        # client_35 = AzureOpenAI(
            # azure_endpoint=os.environ.get("AZURE_OPENAI_35_ENDPOINT", None),
            # api_key=os.environ.get("OPENAI_35_API_KEY", None),
            # api_version=os.environ.get("OPENAI_35_API_VERSION", None),
        # )
        # model_35 = os.environ.get("OPENAI_35_DEPLOYMENT_NAME", None)
        # client_40 = AzureOpenAI(
            # azure_endpoint=os.environ.get("AZURE_OPENAI_40_ENDPOINT", None),
            # api_key=os.environ.get("OPENAI_40_API_KEY", None),
            # api_version=os.environ.get("OPENAI_40_API_VERSION", None),
        # )
        # model_40 = os.environ.get("OPENAI_40_DEPLOYMENT_NAME", None)
        client_4o = AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
            api_key=os.environ.get("OPENAI_4O_API_KEY", None),
            api_version="2024-12-01-preview", # os.environ.get("OPENAI_4O_API_VERSION", None),
        )
        model_4o = "o1" # os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None)
        res = client_4o.chat.completions.create(
            model=model_4o,
            messages=[{"role": "user", "content": "Hi"}],
            # temperature=0,
            max_completion_tokens=4096,
            # top_p=0.95,
            # frequency_penalty=0,
            # presence_penalty=0,
            stop=None,
            timeout=600,
        )

        print(res)


