from extractor.agents.pk_specimen_summary.pk_spec_sum_common_step import PKSpecSumCommonAgentStep
from extractor.agents.pk_specimen_summary.pk_spec_sum_common_agent import (
    PKSpecSumCommonAgentResult,
)
from extractor.agents.pk_specimen_summary.pk_spec_sum_workflow_utils import PKSpecSumWorkflowState

from extractor.agents.pk_specimen_summary.pk_spec_sum_patient_info_refine_agent import (
    PatientInfoRefinedResult,
    get_patient_info_refine_prompt,
    post_process_refined_patient_info,
)


class PatientInfoRefinementStep(PKSpecSumCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Refining Population Information"
        self.end_title = "Completed to Refine Population Information"

    def get_system_prompt(self, state: PKSpecSumWorkflowState):
        title = state["title"]
        full_text = state["full_text"]
        md_table_specimen = state["md_table_specimen"]
        previous_errors = state["previous_errors"] if "previous_errors" in state else "N/A"
        system_prompt = get_patient_info_refine_prompt(title, full_text, md_table_specimen, previous_errors)
        previous_errors_prompt = self._get_previous_errors_prompt(state)
        return system_prompt + previous_errors_prompt

    def leave_step(
        self,
        state: PKSpecSumWorkflowState,
        res: PKSpecSumCommonAgentResult,
        processed_res=None,
        token_usage=None,
    ):
        if processed_res is not None:
            state["md_table_patient_refined"] = processed_res
            self._step_output(state, step_output="Result (md_table_patient_refined):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, res, processed_res, token_usage)

    def get_schema(self):
        return PatientInfoRefinedResult

    def get_post_processor_and_kwargs(self, state: PKSpecSumWorkflowState):
        md_table_specimen = state["md_table_specimen"]
        return post_process_refined_patient_info, {"md_table_specimen": md_table_specimen}
