from langchain_google_genai import ChatGoogleGenerativeAI
import os
import logging

logger = logging.getLogger(__name__)


def get_gemini():
    return ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("GEMINI_MODEL"),  # ("GEMINI_20_MODEL"),
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
