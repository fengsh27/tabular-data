from typing import List, Any, Dict
# from openai import AzureOpenAI, OpenAI
import google.generativeai as genai
from google.generativeai import GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import logging

logger = logging.getLogger(__name__)

if "GEMINI_15_API_KEY" in os.environ:
    genai.configure(api_key=os.environ.get("GEMINI_15_API_KEY", None))

def get_client():
    try:
        model_15_pro = genai.GenerativeModel(
            os.environ.get("GEMINI_15_MODEL", "gemini-pro")
            ) \
            if "GEMINI_15_MODEL" in os.environ else None
        model_15_flash = genai.GenerativeModel(
            os.environ.get("GEMINI_15_FLASH_MODEL", "gemini-1.5-flash-latest")
            ) \
            if "GEMINI_15_FLASH_MODEL" in os.environ else None
        return (model_15_pro, model_15_flash)
    except Exception:
        return (None, None)

def add_message_message_list(msgs: List[Any], msg: Dict[str, Any]):
    cnt = len(msgs)
    if cnt == 0 or msgs[cnt-1]['role'] != msg['role']:
        msgs.append(msg)
        return
    # combine msg
    cur_msg = msg['parts'] if type(msg) is list else [msg['parts']]
    prev_msg = msgs[cnt-1]['parts'] if type(msgs[cnt-1]['parts']) == list else [msgs[cnt-1]['parts']]
    msgs[cnt-1]['parts'] = prev_msg + cur_msg

def convert_messages(messages: List[Any]):
    converted_msgs = []
    for msg in messages:
        if msg["role"] == "system":
            converted_msgs.append({'role': "user", "parts": msg["content"]})
        elif msg['role'] == 'assistant':
            converted_msgs.append({'role': "model", "parts": msg["content"]})
        else:
            converted_msgs.append({'role': msg['role'], 'parts': msg['content']})
    res_msgs = []
    for msg in converted_msgs:
        add_message_message_list(res_msgs, msg)
        
    return res_msgs

def messageDecor(func):
    def converter(*args, **kwargs):
        if len(args) > 0:
            the_args = [*args]
            the_args[0] = convert_messages(args[0])
            args = tuple(the_args)
        else:
            kwargs["messages"] = convert_messages(kwargs["messages"])
        return func(*args, **kwargs)
    return converter

def request_to_gemini(model: GenerativeModel, messages: List[any]):
    res = model.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0,
            # max_output_tokens=10000,
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        request_options={"timeout": 60000}
    )
    usage = (
        model.count_tokens(res.text).total_tokens + 
        model.count_tokens(messages).total_tokens
        if res is not None and res.text is not None else 0
    )
    return (True, res.text, usage, False)

@messageDecor
def request_to_gemini_15_pro(messages: List[Any], question: str):
    add_message_message_list(messages, {"role": "user", "parts": question})
    try:
        model_15_pro, _ = get_client()
        return request_to_gemini(model_15_pro, messages)
    except Exception as e:
        logger.error(e)
        return (False, str(e), None)
    
@messageDecor
def request_to_gemini_15_flash(messages: List[Any], question: str):
    add_message_message_list(messages, {"role": "user", "parts": question})
    try:
        _, model_15_flash = get_client()
        return request_to_gemini(model_15_flash, messages)
    except Exception as e:
        logger.error(e)
        return (False, str(e), None)

