
import os
from langchain_ollama import ChatOllama


def get_gpt_oss():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        # base_url="http://localhost:11434",
        base_url=base_url,
        model="gpt-oss:20b",
        streaming=False,
        model_kwargs={
            "num_ctx": 8192,
            "think": False,
            "stream": False,
            "temperature": 0.0,           # deterministic extraction
            "top_p": 1.0,
            "top_k": 1,
        }
    )

def get_gpt_qwen_235b():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        # base_url="http://localhost:11434",
        base_url=base_url,
        # model="qwen3:235b",
        model="qwen3:235b",
        model_kwargs={
            "num_ctx": 8192,
            "think": False,
            "temperature": 0.0,           # deterministic extraction
            "top_p": 1.0,
            "top_k": 1,
        }
    )

def get_gpt_qwen_30b():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        # base_url="http://localhost:11434",
        base_url=base_url,
        model="qwen3:30b",
        model_kwargs={
            "num_ctx": 8192,
            "think": False,
            "stream": False,
            "temperature": 0.0,           # deterministic extraction
            "top_p": 1.0,
            "top_k": 1,
        }
    )