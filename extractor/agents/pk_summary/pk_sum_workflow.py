
from typing import Optional
from langchain_openai.chat_models.base import BaseChatOpenAI

from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_summary.pk_sum_common_agent import (
    PKSumCommonAgent
)
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.pk_summary.pk_sum_drug_info_agent import (
    DRUG_INFO_PROMPT,
    DrugInfoResult,
    post_process_drug_info,
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

class PKSumWorkflow:
    """ pk summary workflow """
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def build(self):
        pass

        

        

            



