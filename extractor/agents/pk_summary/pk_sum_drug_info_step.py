

from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.agent_utils import (
    display_md_table, 
)
from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonAgentStep
from extractor.agents.pk_summary.pk_sum_common_agent import (
    PKSumCommonAgentResult,
)
from extractor.agents.pk_summary.pk_sum_drug_info_agent import (
    DRUG_INFO_PROMPT, 
    DrugInfoResult, 
    post_process_drug_info,
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

class DrugInfoExtractionStep(PKSumCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Extracting Drug Information"
        self.end_title = "Completed to Extract Drug Information"

    def get_system_prompt(self, state):
        md_table = state["md_table"]
        caption = state["caption"]
        return DRUG_INFO_PROMPT.format(
            processed_md_table = display_md_table(md_table),
            caption=caption,
        )
    
    def leave_step(
        self, 
        state: PKSumWorkflowState, 
        res: PKSumCommonAgentResult, 
        processed_res = None, 
        token_usage = None
    ):
        if processed_res is not None:
            state["md_table_drug"] = processed_res
        super().leave_step(state, res, processed_res, token_usage)
    
    def get_schema(self):
        return DrugInfoResult
    
    def get_post_processor_and_kwargs(self, state):
        return post_process_drug_info, None
            
