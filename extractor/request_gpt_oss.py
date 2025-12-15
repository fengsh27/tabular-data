
import os
from langchain_ollama import ChatOllama

MAX_CONTENT_NUM=16384

def get_gpt_oss():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        # base_url="http://localhost:11434",
        base_url=base_url,
        model="gpt-oss:20b",
        reasoning=False,
        streaming=False,
        num_ctx=MAX_CONTENT_NUM,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
    )

def get_gpt_qwen_235b():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        # base_url="http://localhost:11434",
        base_url=base_url,
        # model="qwen3:235b",
        model="qwen3:235b",
        reasoning=False,
        streaming=False,
        num_ctx=MAX_CONTENT_NUM,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
    )

def get_gpt_qwen_30b(schema: dict | None = None):
    base_url = os.getenv("OLLAMA_BASE_URL")
    if schema is None:
        return ChatOllama(
            # base_url="http://localhost:11434",
            base_url=base_url,
            model="qwen3:30b",
            reasoning=False,
            streaming=False,
            num_ctx=MAX_CONTENT_NUM,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )
    else:
        return ChatOllama(
            # base_url="http://localhost:11434",
            base_url=base_url,
            model="qwen3:30b",
            reasoning=False,
            streaming=False,
            num_ctx=MAX_CONTENT_NUM,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            format=schema,
        )