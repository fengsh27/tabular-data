
from typing import Callable, Optional, TypedDict
from langchain_openai.chat_models.base import BaseChatOpenAI

from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_sum_common_agent import (
    PKSumCommonAgent
)
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.pk_sum_drug_info_agent import (
    DRUG_INFO_PROMPT,
    DrugInfoResult,
    post_process_drug_info,
)

class PKSumWorkflowState(TypedDict):
    """ state data """
    md_table: str
    caption: str
    md_table_drug: Optional[str] = None
    step_callback: Optional[Callable] = None

class PKSumWorkflow:
    """ pk summary workflow """
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def build(self):
        
        def _extract_drug_info(state: PKSumWorkflowState):
            md_table = state.md_table
            caption = state.caption
            agent = PKSumCommonAgent(llm=self.llm)
            _, md_table_dug, token_usage = agent.go(
                system_prompt=DRUG_INFO_PROMPT.format(
                    processed_md_table=display_md_table(md_table),
                    caption=caption,
                ),
                instruction_prompt=INSTRUCTION_PROMPT,
                schema=DrugInfoResult,
                post_process=post_process_drug_info,
            )
            state["md_table_drug"] = md_table_dug

            



