import os
import pytest
from dotenv import load_dotenv
import logging

from .common import LLMClient

load_dotenv()


class ClaudeClient(LLMClient):
    def __init__(self):
        super().__init__()
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))

    def create(self, system_prompts: str, user_prompts: str):
        res = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0.0,
            system=system_prompts,
            messages=[{"role": "user", "content": user_prompts}],
        )
        return (res.content[0].text, 0)


class GptClient(LLMClient):
    def __init__(self):
        super().__init__()
        # from openai import AzureOpenAI
        from langchain_openai import AzureChatOpenAI

        self.client = AzureChatOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
            api_key=os.environ.get("OPENAI_4O_API_KEY", None),
            api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
            azure_deployment=os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None),
            model=os.environ.get("OPENAI_4O_MODEL", None),
            temperature=0.0,
            max_tokens=4096,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )

    def create(self, system_prompts: str, user_prompts: str):
        prompts = [
            {"role": "system", "content": system_prompts},
            {"role": "user", "content": user_prompts},
        ]
        response = self.client.generate(
            messages=[prompts],
        )
        msg = response.generations[0][0].text
        token_usage = response.llm_output.get("token_usage")
        total_usage = token_usage.get("total_tokens", 0)
        return (msg, total_usage)


class GeminiClient(LLMClient):
    def __init__(self):
        super().__init__()
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get("GEMINI_15_API_KEY", None))
        self.client = genai.GenerativeModel(
            os.environ.get("GEMINI_15_MODEL", "gemini-pro")
        )

    def create(self, system_prompts: str, user_prompts: str):
        msgs = [
            {"role": "user", "parts": [system_prompts, user_prompts]},
        ]
        res = self.client.generate_content(
            msgs,
            generation_config=genai.GenerationConfig(
                candidate_count=1,
                temperature=0,
                max_output_tokens=10000,
            ),
        )
        usage = (
            self.client.count_tokens(res.text).total_tokens
            + self.client.count_tokens(msgs).total_tokens
            if res is not None and res.text is not None
            else 0
        )
        return (res.text, usage)


@pytest.fixture(scope="module")
def client():
    return GptClient()  # GptClient, GeminiClient and ClaudeClient are available


@pytest.fixture(scope="session", autouse=True)
def prepare_logging():
    from extractor.log_utils import initialize_logger
    initialize_logger(
        log_file="benchmark.log",
        app_log_name="benchmar",
        app_log_level=logging.INFO,
        log_entries={
            "benchmark": logging.INFO,
        }
    )

