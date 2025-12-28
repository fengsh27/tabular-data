from langchain_openai.chat_models.base import BaseChatOpenAI

from extractor.agents.common_agent.common_agent_2steps import CommonAgentTwoSteps
from extractor.agents.common_agent.common_agent import CommonAgent
from extractor.agents.common_agent.common_agent_ollama import (
    CommonAgentOllama, 
    CommonAgentOllamaTwoChainSteps,
    CommonAgentOllamaTwoSteps,
)

from extractor.request_gpt_oss import get_gpt_oss, get_gpt_qwen_30b
from extractor.request_openai import get_openai, get_5_openai


def get_pipeline_llm():
    return get_gpt_qwen_30b() # get_gpt_oss() # get_openai() # 

def get_agent_llm():
    return get_gpt_qwen_30b() # get_gpt_oss() # get_5_openai() # 


def get_common_agent(llm: BaseChatOpenAI):
    # return CommonAgentTwoSteps(llm=llm)
    # return CommonAgent(llm=llm)
    return CommonAgentOllama(llm=llm)
    # return CommonAgentOllamaTwoChainSteps(llm=llm)
