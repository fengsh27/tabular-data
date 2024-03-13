from typing import List, Any
from openai import AzureOpenAI, OpenAI
import os
import logging

from src.article_stamper import Stamper

logger = logging.getLogger(__name__)

openai_type = os.environ.get("OPENAI_API_TYPE")
if openai_type == "azure":
    client = AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        api_key=os.environ.get("OPENAI_API_KEY", None),
        api_version=os.environ.get("OPENAI_API_VERSION", None),
    )
    model = os.environ.get("OPENAI_DEPLOYMENT_NAME", None)
else:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))
    model = "gpt-3.5-turbo" # "gpt-4-1106-preview"

def request_to_chatgpt(prompts: List[Any], question: str, stamper: Stamper):
    prompts.append({"role": "user", "content": question})
    stamper.output_prompts(prompts)
    try:
        res = client.chat.completions.create(
            model=model, 
            messages=prompts,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        try:
            stamper.output_result(res.choices[0].message.content)
        except Exception as e:
            logger.warn(e)
        return (True, res.choices[0], res.usage)
    except Exception as e:
        logger.error(e)
        return (False, str(e))

