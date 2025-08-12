
import os
from langchain_meta import ChatMetaLlama

def get_meta_llama():
    api_key = os.environ.get("LLAMA_API_KEY")
    model = os.environ.get("LLAMA_MODEL")
    llm = ChatMetaLlama(
        model=model,
        api_key=api_key,
        base_url="https://api.llama.com/v1/",
        temperature=0,
        max_tokens=64000,
    )
    return llm
