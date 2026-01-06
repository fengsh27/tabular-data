import os
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_ollama.chat_models import ChatOllama

from extractor.agents.common_agent.common_agent_2steps import CommonAgentTwoSteps
from extractor.agents.common_agent.common_agent import CommonAgent
from extractor.agents.common_agent.common_agent_ollama import (
    CommonAgentOllama, 
    CommonAgentOllamaTwoChainSteps,
    CommonAgentOllamaTwoSteps,
)

from extractor.request_gpt_oss import get_gpt_oss, get_gpt_qwen_30b
from extractor.request_openai import get_openai, get_5_openai

MAX_PIPELINE_AGENT_CONTENT_NUM = 16384
MAX_PIPELINE_AGENT_PREDICT_NUM = 8192

def get_pipeline_llm():
    llm = os.getenv("PIPELINE_LLM", "GPT-OSS-20B")
    if llm == "GPT-OSS-20B":
        return get_gpt_oss(
            max_content_num=MAX_PIPELINE_AGENT_CONTENT_NUM,
            max_predict_num=MAX_PIPELINE_AGENT_PREDICT_NUM,
        )
    elif llm == "QWEN3-30B":
        return get_gpt_qwen_30b(
            max_content_num=MAX_PIPELINE_AGENT_CONTENT_NUM,
            max_predict_num=MAX_PIPELINE_AGENT_PREDICT_NUM,
        )
    elif llm == "OPENAI":
        return get_openai()
    elif llm == "OPENAI-5":
        return get_5_openai()
    else:
        raise ValueError(f"Unknown LLM: {llm}")

def get_agent_llm():
    llm = os.getenv("AGENT_LLM", "GPT-OSS-20B")
    if llm == "GPT-OSS-20B":
        return get_gpt_oss()
    elif llm == "QWEN3-30B":
        return get_gpt_qwen_30b()
    elif llm == "OPENAI":
        return get_openai()
    elif llm == "OPENAI-5":
        return get_5_openai()
    else:
        raise ValueError(f"Unknown LLM: {llm}")


def get_common_agent(llm: BaseChatOpenAI, two_steps: bool = False):
    # return CommonAgentTwoSteps(llm=llm)
    # return CommonAgent(llm=llm)
    if isinstance(llm, ChatOllama):
        return CommonAgentOllama(llm=llm)
    if two_steps:
        return CommonAgentTwoSteps(llm=llm)
    return CommonAgent(llm=llm)
