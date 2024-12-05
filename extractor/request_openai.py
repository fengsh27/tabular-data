from typing import List, Any, Optional
from openai import AzureOpenAI, OpenAI
import os
import logging

from extractor.utils import concate_llm_contents

logger = logging.getLogger(__name__)

def get_client_and_model(): 
    """
    Obtain GPT client and model
    Return:
        (gpt-4o, model-4o, gpt-35, model-35, gpt-40, model-40)
    """
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
            api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        )
        model_4o = os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None)
        return (client_4o, model_4o, None, None, None, None)
    
    else:
        # client_35 = OpenAI(api_key=os.environ.get("OPENAI_35_API_KEY", None))
        # model_35 = "gpt-3.5-turbo" # "gpt-4-1106-preview"
        # client_40 = OpenAI(api_key=os.environ.get("OPENAI_40_API_KEY", None))
        # model_40 = "gpt-4-1106-preview"
        client_4o = OpenAI(api_key=os.environ.get("OPENAI_4O_API_KEY", None))
        model_4o = "gpt-4o"
        return (client_4o, model_4o, None, None, None, None)



def request_to_chatgpt_35(prompts: List[Any], question: str):
    prompts.append({"role": "user", "content": question})
    try:
        _, _, client_35, model_35, _, _ = get_client_and_model()
        res = client_35.chat.completions.create(
            model=model_35, 
            messages=prompts,
            temperature=0,
            # max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            timeout=600,
        )
        usage = res.usage.total_tokens
        return (True, res.choices[0].message.content, usage, False)
    except Exception as e:
        logger.error(e)
        return (False, str(e), None, False)

def request_to_chatgpt_40(prompts: List[Any], question: str):
    prompts.append({"role": "user", "content": question})
    try:
        _, _, _, _, client_40, model_40 = get_client_and_model()
        res = client_40.chat.completions.create(
            model=model_40,
            messages=prompts,
            temperature=0,
            max_tokens=20000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            timeout=600, # in seconds, 10 mins
        )
        usage = res.usage.total_tokens
        return (True, res.choices[0].message.content, usage, False)
    except Exception as e:
        logger.error(e)
        return (False, str(e), None, False)
    
def request_to_chatgpt_4o(prompts: List[Any], question: str):
    prompts.append({"role": "user", "content": question})
    try:
        client_4o, model_4o, _, _, _, _ = get_client_and_model()
        res = client_4o.chat.completions.create(
            model=model_4o,
            messages=prompts,
            temperature=0,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            timeout=600,
        )
        content = res.choices[0].message.content
        usage = res.usage.total_tokens

        if not _is_incompleted_response(content):
            return (True, content, usage, False)
        
        contents = [content]
        usages = [usage]
        loops = 0
        MAX_LOOP = 5
        while _is_incompleted_response(content) and loops < MAX_LOOP:
            if content is not None:
                prompts.append({"role": "assistant", "content": content})
            prompts.append({"role": "user", "content": "the output json table is not completed, please continue to generate the json table without any other text."})
            res = client_4o.chat.completions.create(
                model=model_4o,
                messages=prompts,
                temperature=0,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                timeout=600,
            )
            content = res.choices[0].message.content
            usage = res.usage.total_tokens
            contents.append(content)
            usages.append(usage)
            loops += 1
        
        all_content, all_usage, truncated = concate_llm_contents(contents, usages)
        return (True, all_content, all_usage, True)
    except Exception as e:
        logger.error(e)
        return (False, str(e), None, False)
    
def _is_incompleted_response(content: Optional[str] = None):
    if content is None:
        return False
    stripped_content = content.strip()
    if not stripped_content.endswith("}]") \
       and not stripped_content.endswith("```"):
        return True
    return False
