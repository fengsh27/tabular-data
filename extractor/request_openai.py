from typing import List, Any, Optional
# from openai import AzureOpenAI, OpenAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import openai
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
        # client_35 = AzureChatOpenAI(
            # api_key=os.environ.get("OPENAI_35_API_KEY", None),
            # azure_endpoint=os.environ.get("AZURE_OPENAI_35_ENDPOINT", None),
            # api_version=os.environ.get("OPENAI_35_API_VERSION", None),
            # azure_deployment=os.environ.get("OPENAI_35_DEPLOYMENT_NAME", None),
            # model=os.environ.get("OPENAI_35_MODEL", None),
        # )
        # model_35 = os.environ.get("OPENAI_35_DEPLOYMENT_NAME", None)
        # client_40 = AzureChatOpenAI(
            # api_key=os.environ.get("OPENAI_40_API_KEY", None),
            # azure_endpoint=os.environ.get("AZURE_OPENAI_40_ENDPOINT", None),
            # api_version=os.environ.get("OPENAI_40_API_VERSION", None),
            # azure_deployment=os.environ.get("OPENAI_40_DEPLOYMENT_NAME", None),
            # model=os.environ.get("OPENAI_40_MODEL", None),
        # )
        # model_40 = os.environ.get("OPENAI_40_DEPLOYMENT_NAME", None)
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
        # client_35 = OpenAI(api_key=os.environ.get("OPENAI_35_API_KEY", None))
        # model_35 = "gpt-3.5-turbo" # "gpt-4-1106-preview"
        # client_40 = OpenAI(api_key=os.environ.get("OPENAI_40_API_KEY", None))
        # model_40 = "gpt-4-1106-preview"
        client_4o = ChatOpenAI(
            api_key=os.environ.get("OPENAI_4O_API_KEY", None),
            model=os.environ.get("OPENAI_4O_MODEL", None),
        )
        model_4o = "gpt-4o"
        return (client_4o, model_4o, None, None, None, None)


def request_to_chatgpt_4o(prompts: List[Any], question: str):
    prompts.append({"role": "user", "content": question})
    try:
        client_4o, model_4o, _, _, _, _ = get_client_and_model()
        try:
            res = client_4o.generate(
                messages=[prompts],
            )
        except (
            openai._exceptions.APIError,
            openai._exceptions.OpenAIError,
            openai._exceptions.ConflictError,
            openai._exceptions.NotFoundError,
            openai._exceptions.APIStatusError,
            openai._exceptions.RateLimitError,
            openai._exceptions.APITimeoutError,
            openai._exceptions.BadRequestError,
            openai._exceptions.APIConnectionError,
            openai._exceptions.AuthenticationError,
            openai._exceptions.InternalServerError,
            openai._exceptions.PermissionDeniedError,
            openai._exceptions.UnprocessableEntityError,
            openai._exceptions.APIResponseValidationError,
        ) as e:
            return False, str(e), None, None
        except Exception as e:
            return False, str(e), None, None
        content = res.generations[0][0].text
        token_usage = res.llm_output.get("token_usage")
        total_tokens = token_usage.get("total_tokens", 0)

        if not _is_incompleted_response(content):
            return (True, content, total_tokens, False)
        
        contents = [content]
        usages = [total_tokens]
        loops = 0
        MAX_LOOP = 5
        while _is_incompleted_response(content) and loops < MAX_LOOP:
            if content is not None:
                prompts.append({"role": "assistant", "content": content})
            prompts.append({"role": "user", "content": "The JSON table above is incomplete. Continue generating the remaining JSON table content without adding any explanations, comments, or extra text—only the JSON data."})
            res = client_4o.generate(messages=[prompts])
            content = res.generations[0][0].text
            token_usage = res.llm_output.get("token_usage")
            total_tokens = token_usage.get("total_tokens", 0)
            contents.append(content)
            usages.append(total_tokens)
            loops += 1
        
        all_content, all_usage, truncated = concate_llm_contents(contents, usages)
        return (True, all_content, all_usage, True)
    except Exception as e:
        logger.error(e)
        return (False, str(e), None, False)
    
def _is_incompleted_response(content: Optional[str] = None):
    if content is None:
        return False
    max_tokens = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096)) - 10
    stripped_content = content.strip()
    if len(content) > max_tokens and \
        not stripped_content.endswith("}]") \
        and not stripped_content.endswith("```"):
        return True
    return False
