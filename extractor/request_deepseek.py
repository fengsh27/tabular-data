
from typing import List, Any, Dict
from langchain_deepseek import ChatDeepSeek
import os
import logging

logger = logging.getLogger(__name__)

def get_deepseek():
    return ChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model=os.getenv("DEEPSEEK_MODEL"),
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


