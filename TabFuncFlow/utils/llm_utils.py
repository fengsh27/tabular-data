from typing import Optional, List, Dict, Callable
from extractor.request_openai import (
    request_to_chatgpt_35,
    request_to_chatgpt_40,
    request_to_chatgpt_4o,
)
from extractor.request_geminiai import (
    request_to_gemini_15_pro,
    request_to_gemini_15_flash,
)
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
if "GEMINI_15_API_KEY" in os.environ:
    genai.configure(api_key=os.environ.get("GEMINI_15_API_KEY", None))


def get_llm_response(messages, question, model="gemini_15_pro"):
    """
    A further wrapper around Shaohong's request_llm function.
    Send messages and question to the specified LLM and return response details.
    """

    prompt_list = [{"role": "user", "content": msg} for msg in messages]

    if model == "chatgpt_4o":
        request_llm = request_to_chatgpt_4o
    elif model == "gemini_15_pro":
        request_llm = request_to_gemini_15_pro
    else:
        raise ValueError(f"Unsupported model: {model}")

    res, content, usage, truncated = request_llm(prompt_list, question)
    return res, content, usage, truncated

