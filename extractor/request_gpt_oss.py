
import os
from typing import Optional
from pydantic import BaseModel
import requests
import tiktoken
from langchain_core.messages import BaseMessage
from tenacity import retry, stop_after_attempt, wait_incrementing

from extractor.llm_utils import get_format_instructions, get_schema_format

MAX_CONTENT_NUM = 16384
MAX_PREDICT_NUM = 8192

def approx_token_count(text: str | list[BaseMessage]) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    if isinstance(text, str):
        return len(enc.encode(text))
    return sum(len(enc.encode(m.content or "")) for m in text)

class OllamaClientResult:
    def __init__(self, content: str, prompt_tokens: int, predict_tokens: int, raw: dict):
        self.content = content
        self.raw = raw
        self.usage_metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": predict_tokens,
            "total_tokens": prompt_tokens + predict_tokens,
            "approximate": True,
            "tokenizer": "tiktoken/cl100k_base",
        }

class OllamaClient:
    def __init__(self, model: str, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model

    def _prepare_messages(self, messages: list[BaseMessage] | str, format_instructions: str | None = None):
        if isinstance(messages, str):
            return messages + f"{("\n\n" + format_instructions) if format_instructions else ""}"
        
        if isinstance(messages, list):
            if len(messages) == 1:
                return messages[0].content + f"{("\n\n" + format_instructions) if format_instructions else ""}"
            
            ret_msg = messages[0].content + f"{("\n\n" + format_instructions) if format_instructions else ""}"
            for i in range(1, len(messages)):
                ret_msg += "\n\n" + messages[i].content
            return ret_msg


    def _think_value(self):
        # Ollama supports boolean or levels; GPT-OSS often behaves best with a level. :contentReference[oaicite:2]{index=2}
        if self.model.startswith("gpt-oss"):
            return "low"   # try "low" first; if you truly want none, see notes below
        return False

    def _prepare_payload(
        self,
        messages: list[BaseMessage] | str,
        schema: Optional[BaseModel | dict] = None,
    ) -> dict:
        if self.model.startswith("gpt-oss"):
            format_instructions = get_format_instructions(schema)
            prepared_message = self._prepare_messages(messages, format_instructions)
            return {
                "model": self.model,
                "prompt": prepared_message,
                "stream": False,
                "think": self._think_value(),  # NOTE: 'think' is top-level for /api/chat in Ollama docs. :contentReference[oaicite:3]{index=3}
                "options": {
                    "num_ctx": MAX_CONTENT_NUM,
                    "num_predict": MAX_PREDICT_NUM,
                    "temperature": 0.7,
                },
            }
        format_dict = get_schema_format(schema)
        return {
            "model": self.model,
            "prompt": self._prepare_messages(messages),
            "stream": False,
            "think": self._think_value(),  # NOTE: 'think' is top-level for /api/chat in Ollama docs. :contentReference[oaicite:3]{index=3}
            "format": format_dict,
            "options": {
                "num_ctx": MAX_CONTENT_NUM,
                "num_predict": MAX_PREDICT_NUM,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
            },
        }

    @retry(stop=stop_after_attempt(2), wait=wait_incrementing(start=1.0, increment=1, max=5))
    def _request_ollama(self, messages: list[BaseMessage] | str, schema: Optional[BaseModel | dict] = None):
        payload = self._prepare_payload(messages, schema)
        # {
        #     "model": self.model,
        #     "prompt": prepared_messages,
        #     "stream": False,
        #     "think": self._think_value(),  # NOTE: 'think' is top-level for /api/chat in Ollama docs. :contentReference[oaicite:3]{index=3}
        #     "options": {
        #         "num_ctx": MAX_CONTENT_NUM,
        #         "num_predict": 8192,
        #         "temperature": 0.0,
        #         "top_p": 1.0,
        #         "top_k": 1,
        #     },
        # }

        r = requests.post(f"{self.base_url}/api/generate", json=payload)
        res = r.json()
        return res.get("response", "")

    def invoke(self, messages: list[BaseMessage] | str, schema: Optional[BaseModel | dict] = None):
        prompt_tokens = approx_token_count(messages)

        res = self._request_ollama(messages, schema)

        content = res

        predict_tokens = approx_token_count(content)
        return OllamaClientResult(content, prompt_tokens, predict_tokens, raw=res)


def get_gpt_oss():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return OllamaClient("gpt-oss:20b", base_url)

def get_gpt_qwen_235b():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return OllamaClient("qwen3:235b", base_url)

def get_gpt_qwen_30b():
    base_url = os.getenv("OLLAMA_BASE_URL")
    return OllamaClient("qwen3:30b", base_url)

