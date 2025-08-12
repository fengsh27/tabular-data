from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_individual.pk_ind_common_step import PKIndCommonAgentStep
from extractor.agents.pk_individual.pk_ind_summary_data_del_agent import (
    SUMMARY_DATA_DEL_PROMPT,
    SummaryDataDelResult,
    post_process_summary_del_result,
)


class SummaryDataDelStep(PKIndCommonAgentStep):
    """The step to delete individual data"""

    def __init__(self):
        super().__init__()
        self.start_title = "Deleting Summary Data"
        self.end_title = "Completed to Deleting Summary Data"

    def get_system_prompt(self, state):
        md_table = state["md_table"]
        system_prompt = SUMMARY_DATA_DEL_PROMPT.format(
            processed_md_table=display_md_table(md_table)
        )
        previous_errors_prompt = self._get_previous_errors_prompt(state)
        return system_prompt + previous_errors_prompt

    def get_schema(self):
        return SummaryDataDelResult

    def get_post_processor_and_kwargs(self, state):
        md_table = state["md_table"]
        return post_process_summary_del_result, {"md_table": md_table}

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["md_table_individual"] = processed_res
            self._step_output(state, step_output="Result (md_table_individual):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, res, processed_res, token_usage)
