
import os
from langchain_ollama import ChatOllama


def get_gpt_oss():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        # base_url="http://localhost:11434",
        base_url=base_url,
        model="gpt-oss:20b",
        temperature=0,
        max_tokens=16384,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )

def get_gpt_qwen_235b():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        # base_url="http://localhost:11434",
        base_url=base_url,
        # model="qwen3:235b",
        model="qwen3:235b",
        temperature=0,
        max_tokens=16384,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )

def get_gpt_qwen_30b():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        # base_url="http://localhost:11434",
        base_url=base_url,
        model="qwen3:30b",
        temperature=0,
        max_tokens=16384,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )