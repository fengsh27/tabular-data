
import os
from openai import AzureOpenAI
import google.generativeai as genai
from google.generativeai import GenerativeModel
import anthropic
import pytest
from dotenv import load_dotenv
load_dotenv()

class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
        
    def create(self, system_prompts: str, user_prompts: str):
        res = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0.0,
            system=system_prompts,
            messages=[
                {"role": "user", "content": user_prompts}
            ]        
        )
        return (res.content[0].text, 0)
    
class GptClient:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
            api_key=os.environ.get("OPENAI_4O_API_KEY", None),
            api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        )
        self.model_4o = os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None)
    def create(self, system_prompts: str, user_prompts: str):
        prompts = [
            {"role": "system", "content": system_prompts},
            {"role": "user", "content": user_prompts},
        ]
        res = self.client.chat.completions.create(
            model=self.model_4o,
            temperature=0.0,
            max_tokens=4096,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            messages=prompts,
        )
        return (res.choices[0].message.content, res.usage.total_tokens)

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=os.environ.get("GEMINI_15_API_KEY", None))
        self.client =genai.GenerativeModel(os.environ.get("GEMINI_15_MODEL", "gemini-pro"))

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
            self.client.count_tokens(res.text).total_tokens + 
            self.client.count_tokens(msgs).total_tokens
            if res is not None and res.text is not None else 0
        )
        return (res.text, usage)

@pytest.fixture(scope="module")
def client():    
    return ClaudeClient() # GptClient, GeminiClient and ClaudeClient are available

