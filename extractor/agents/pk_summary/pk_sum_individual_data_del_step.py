
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonStep
from extractor.agents.pk_summary.pk_sum_individual_data_del_agent import (
    INDIVIDUAL_DATA_DEL_PROMPT,
    IndividualDataDelResult,
    post_process_individual_del_result,
)

class IndividualDataDelStep(PKSumCommonStep):
    """ The step to delete individual data """
    def __init__(self):
        super().__init__()
        self.start_title = "Deleting Individual Data"
        self.end_title = "Completed to Deleting Individual Data"

    def get_system_prompt(self, state):
        md_table = state["md_table"]
        return INDIVIDUAL_DATA_DEL_PROMPT.format(
            processed_md_table=display_md_table(md_table)
        )
    
    def get_schema(self):
        return IndividualDataDelResult
    
    def get_post_processor_and_kwargs(self, state):
        md_table = state["md_table"]
        return post_process_individual_del_result, {"md_table": md_table}
    
    def leave_step(self, state, res, processed_res = None, token_usage = None):
        if processed_res is not None:
            state["md_table_summary"] = processed_res
        return super().leave_step(state, res, processed_res, token_usage)
