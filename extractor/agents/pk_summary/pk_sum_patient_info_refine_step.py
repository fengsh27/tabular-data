from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonAgentStep
from extractor.agents.pk_summary.pk_sum_common_agent import (
    PKSumCommonAgentResult,
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

from extractor.agents.pk_summary.pk_sum_patient_info_refine_agent import (
    PatientInfoRefinedResult,
    get_patient_info_refine_prompt,
    post_process_refined_patient_info,
)


class PatientInfoRefinementStep(PKSumCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Refining Population Information"
        self.end_title = "Completed to Refine Population Information"

    def get_system_prompt(self, state: PKSumWorkflowState):
        md_table = state["md_table"]
        md_table_patient = state["md_table_patient"]
        caption = state["caption"]
        return get_patient_info_refine_prompt(md_table, md_table_patient, caption)

    def leave_step(
        self,
        state: PKSumWorkflowState,
        step_reasoning_process: str=None,
        processed_res=None,
        token_usage=None,
    ):
        if processed_res is not None:
            state["md_table_patient_refined"] = processed_res
            self._step_output(state, step_output="Result (md_table_patient_refined):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, step_reasoning_process, processed_res, token_usage)

    def get_schema(self):
        return PatientInfoRefinedResult

    def get_post_processor_and_kwargs(self, state: PKSumWorkflowState):
        md_table_patient = state["md_table_patient"]
        return post_process_refined_patient_info, {"md_table_patient": md_table_patient}
