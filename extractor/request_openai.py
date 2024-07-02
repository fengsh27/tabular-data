from typing import List, Any
from openai import AzureOpenAI, OpenAI
import os
import logging

logger = logging.getLogger(__name__)

openai_type = os.environ.get("OPENAI_API_TYPE")
if openai_type == "azure":
    client_35 = AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_35_ENDPOINT", None),
        api_key=os.environ.get("OPENAI_35_API_KEY", None),
        api_version=os.environ.get("OPENAI_35_API_VERSION", None),
    )
    model_35 = os.environ.get("OPENAI_35_DEPLOYMENT_NAME", None)
    client_40 = AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_40_ENDPOINT", None),
        api_key=os.environ.get("OPENAI_40_API_KEY", None),
        api_version=os.environ.get("OPENAI_40_API_VERSION", None),
    )
    model_40 = os.environ.get("OPENAI_40_DEPLOYMENT_NAME", None)
    client_4o = AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
        api_key=os.environ.get("OPENAI_4O_API_KEY", None),
        api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
    )
    model_4o = os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None)

else:
    client_35 = OpenAI(api_key=os.environ.get("OPENAI_35_API_KEY", None))
    model_35 = "gpt-3.5-turbo" # "gpt-4-1106-preview"
    client_40 = OpenAI(api_key=os.environ.get("OPENAI_40_API_KEY", None))
    model_40 = "gpt-4-1106-preview"
    client_4o = OpenAI(api_key=os.environ.get("OPENAI_4O_API_KEY", None))
    model_4o = "gpt-4o"

def request_to_chatgpt_35(prompts: List[Any], question: str):
    prompts.append({"role": "user", "content": question})
    try:
        res = client_35.chat.completions.create(
            model=model_35, 
            messages=prompts,
            temperature=0,
            # max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        usage = res.usage.total_tokens
        return (True, res.choices[0].message.content, usage)
    except Exception as e:
        logger.error(e)
        return (False, str(e), None)

def request_to_chatgpt_40(prompts: List[Any], question: str):
    prompts.append({"role": "user", "content": question})
    try:
        res = client_40.chat.completions.create(
            model=model_40,
            messages=prompts,
            temperature=0,
            max_tokens=20000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        usage = res.usage.total_tokens
        return (True, res.choices[0].message.content, usage)
    except Exception as e:
        logger.error(e)
        return (False, str(e), None)
    
def request_to_chatgpt_4o(prompts: List[Any], question: str):
    prompts.append({"role": "user", "content": question})
    try:
        res = client_4o.chat.completions.create(
            model=model_4o,
            messages=prompts,
            temperature=0,
            max_tokens=4096,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        usage = res.usage.total_tokens
        return (True, res.choices[0].message.content, usage)
    except Exception as e:
        logger.error(e)
        return (False, str(e), None)
