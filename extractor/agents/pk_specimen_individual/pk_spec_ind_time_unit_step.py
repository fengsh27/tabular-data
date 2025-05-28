from extractor.agents.pk_specimen_individual.pk_spec_ind_common_step import PKSpecIndCommonAgentStep
from extractor.agents.pk_specimen_individual.pk_spec_ind_common_agent import (
    PKSpecIndCommonAgentResult,
)
from extractor.agents.pk_specimen_individual.pk_spec_ind_workflow_utils import PKSpecIndWorkflowState

from extractor.agents.pk_specimen_individual.pk_spec_ind_time_unit_agent import (
    TimeAndUnitResult,
    get_time_and_unit_prompt,
    post_process_time_and_unit,
)


class TimeExtractionStep(PKSpecIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Time Extraction"
        self.end_title = "Completed Time Extraction"

    def get_system_prompt(self, state: PKSpecIndWorkflowState):
        title = state["title"]
        full_text = state["full_text"]
        md_table_specimen = state["md_table_specimen"]
        return get_time_and_unit_prompt(title, full_text, md_table_specimen)

    def leave_step(
        self,
        state: PKSpecIndWorkflowState,
        res: PKSpecIndCommonAgentResult,
        processed_res=None,
        token_usage=None,
    ):
        if processed_res is not None:
            state["md_table_time"] = processed_res
            self._step_output(state, step_output="Result (md_table_time):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, res, processed_res, token_usage)

    def get_schema(self):
        return TimeAndUnitResult

    def get_post_processor_and_kwargs(self, state: PKSpecIndWorkflowState):
        md_table_specimen = state["md_table_specimen"]
        return post_process_time_and_unit, {"md_table_specimen": md_table_specimen}
