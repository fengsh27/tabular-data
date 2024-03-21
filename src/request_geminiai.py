from typing import List, Any, Dict
# from openai import AzureOpenAI, OpenAI
import google.generativeai as genai
import os
import logging

logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", None))
model =genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-pro"))

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

@messageDecor
def request_to_gemini(messages: List[Any], question: str):
    add_message_message_list(messages, {"role": "user", "parts": question})
    try:
        res = model.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=1.0
            )
        )
        usage = (
            model.count_tokens(res.text).total_tokens + 
            model.count_tokens(messages).total_tokens
            if res is not None and res.text is not None else 0
        )
        return (True, res.text, usage)
    except Exception as e:
        logger.error(e)
        return (False, str(e), None)

