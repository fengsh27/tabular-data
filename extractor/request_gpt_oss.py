
from langchain_ollama import ChatOllama


def get_gpt_oss():
    return ChatOllama(
        base_url="http://localhost:11434",
        model="gpt-oss:20b",
        temperature=0,
        max_tokens=16384,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
