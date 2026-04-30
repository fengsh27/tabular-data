
import os
import requests
from typing import Any, Iterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import ChatOllama

MAX_CONTENT_NUM = 16384 * 2
MAX_PREDICT_NUM = 2048 * 2


class ChatGptOss(BaseChatModel):
    """OpenAI-compatible chat model that correctly surfaces content when the
    server also returns a reasoning_content field (e.g. thinking models served
    via vLLM).  LangChain's ChatOpenAI drops content in that case; this class
    calls the endpoint directly via requests and reads the content field."""

    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: int = 300

    @property
    def _llm_type(self) -> str:
        return "gpt-oss"

    def _to_openai_message(self, m: BaseMessage) -> dict:
        if isinstance(m, SystemMessage):
            return {"role": "system", "content": m.content}
        if isinstance(m, AIMessage):
            return {"role": "assistant", "content": m.content}
        return {"role": "user", "content": m.content}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": self.model_name,
            "messages": [self._to_openai_message(m) for m in messages],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop

        resp = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        raw_msg = choice["message"]
        content = raw_msg.get("content") or ""
        reasoning = raw_msg.get("reasoning_content") or ""

        ai_msg = AIMessage(
            content=content,
            additional_kwargs={"reasoning_content": reasoning} if reasoning else {},
            response_metadata={
                "model_name": data.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "token_usage": data.get("usage", {}),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


def get_gpt_oss(
    max_content_num: int = -1,
    max_predict_num: int = -1,
) -> ChatGptOss:
    return ChatGptOss(
        base_url=os.environ.get("OLLAMA_BASE_URL", ""),
        api_key=os.environ.get("GPT_OSS_API_KEY", ""),
        model_name=os.environ.get("GPT_OSS_MODEL", ""),
        temperature=0.1,
        max_tokens=(
            int(os.environ["GPT_OSS_MAX_OUTPUT_TOKENS"])
            if os.environ.get("GPT_OSS_MAX_OUTPUT_TOKENS")
            else None
        ),
        timeout=300,
    )


def get_gpt_qwen_235b(
    max_content_num: int = -1,
    max_predict_num: int = -1,
):
    base_url = os.getenv("OLLAMA_BASE_URL")
    return ChatOllama(
        base_url=base_url,
        model="qwen3:235b",
        reasoning=False,
        streaming=True,
        num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
        num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        timeout=900,
    )


def get_gpt_qwen_30b(
    max_content_num: int = -1,
    max_predict_num: int = -1,
    schema: dict | None = None,
):
    base_url = os.getenv("OLLAMA_BASE_URL")
    if schema is None:
        return ChatOllama(
            base_url=base_url,
            model="qwen3:30b",
            reasoning=False,
            streaming=True,
            num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
            num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            timeout=900,
        )
    else:
        return ChatOllama(
            base_url=base_url,
            model="qwen3:30b",
            reasoning=False,
            streaming=True,
            num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
            num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            format=schema,
            timeout=900,
        )


def get_gpt_qwen35_27b(
    max_content_num: int = -1,
    max_predict_num: int = -1,
    schema: dict | None = None,
):
    base_url = os.getenv("OLLAMA_BASE_URL")
    if schema is None:
        return ChatOllama(
            base_url=base_url,
            model="qwen3.5:27b",
            reasoning=False,
            streaming=True,
            num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
            num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            timeout=900,
        )
    else:
        return ChatOllama(
            base_url=base_url,
            model="qwen3.5:27b",
            reasoning=False,
            streaming=True,
            num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
            num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            format=schema,
            timeout=900,
        )


def get_gemma4_31b(
    max_content_num: int = -1,
    max_predict_num: int = -1,
    schema: dict | None = None,
):
    base_url = os.getenv("OLLAMA_BASE_URL")
    if schema is None:
        return ChatOllama(
            base_url=base_url,
            model="gemma4:31b",
            reasoning=False,
            streaming=True,
            num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
            num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            timeout=900,
        )
    else:
        return ChatOllama(
            base_url=base_url,
            model="gemma4:31b",
            reasoning=False,
            streaming=True,
            num_ctx=max_content_num if max_content_num > 0 else MAX_CONTENT_NUM,
            num_predict=max_predict_num if max_predict_num > 0 else MAX_PREDICT_NUM,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            format=schema,
            timeout=900,
        )
