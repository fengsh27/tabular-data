# from openai import AzureOpenAI, OpenAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import os
import logging


logger = logging.getLogger(__name__)


def get_openai():
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_MODEL", None),
        max_retries=5,
        temperature=0.0,
        max_completion_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096)),
        top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
    )

def get_5_mini_openai():
    MAX_OUTPUT_TOKENS = 16384*5
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_5_MINI_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_5_MINI_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_5_MINI_MODEL", None),
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )

def get_5_openai():
    MAX_OUTPUT_TOKENS = 16384*5
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_5_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_5_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_5_MODEL", None),
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )

def get_client_and_model():
    """
    Obtain GPT client and model
    Return:
        (gpt-4o, model-4o, gpt-35, model-35, gpt-40, model-40)
    """
    openai_type = os.environ.get("OPENAI_API_TYPE")
    if openai_type == "azure":
        client_4o = AzureChatOpenAI(
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
        model_4o = os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None)
        return (client_4o, model_4o, None, None, None, None)

    else:
        client_4o = ChatOpenAI(
            api_key=os.environ.get("OPENAI_4O_API_KEY", None),
            model=os.environ.get("OPENAI_4O_MODEL", None),
        )
        model_4o = "gpt-4o"
        return (client_4o, model_4o, None, None, None, None)
