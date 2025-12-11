from langchain_openai.chat_models.base import BaseChatOpenAI
from extractor.agents.common_agent.common_step import CommonStep
from extractor.agents.agent_factory import get_common_agent

class PKPECommonStep(CommonStep):
    def __init__(self, llm: BaseChatOpenAI):
        super().__init__(llm)

    def get_agent(self, state):
        return get_common_agent(llm=state["llm"])
