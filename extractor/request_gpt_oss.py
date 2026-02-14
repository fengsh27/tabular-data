
import os
from langchain_ollama import ChatOllama

MAX_CONTENT_NUM=16384*6
MAX_PREDICT_NUM=16384*4

def get_gpt_oss(
    max_content_num: int = -1,
    max_predict_num: int = -1,
):
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        # base_url="http://localhost:11434",
        base_url=base_url,
        model="gpt-oss:20b",
        reasoning=False,
        streaming=False,
        num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
        num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
    )

def get_gpt_qwen_235b(
    max_content_num: int = -1,
    max_predict_num: int = -1,
):
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        # base_url="http://localhost:11434",
        base_url=base_url,
        # model="qwen3:235b",
        model="qwen3:235b",
        reasoning=False,
        streaming=False,
        num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
        num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
    )

def get_gpt_qwen_30b(
    max_content_num: int = -1,
    max_predict_num: int = -1,
    schema: dict | None = None,
):
    base_url = os.getenv("OLLAMA_BASE_URL")
    if schema is None:
        return ChatOllama(
            # base_url="http://localhost:11434",
            base_url=base_url,
            model="qwen3:30b",
            reasoning=False,
            streaming=False,
            num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
            num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
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
            num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
            num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            format=schema,
        )