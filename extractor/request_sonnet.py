
import os
from langchain_anthropic import ChatAnthropic


def get_sonnet():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL")
    max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", 64000))
    return ChatAnthropic(
        model=model, 
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=0,
        timeout=None,
        max_retries=2,
    )




