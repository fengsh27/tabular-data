from extractor.agents.pk_drug_summary.pk_drug_sum_common_step import PKDrugSumCommonAgentStep
from extractor.agents.pk_drug_summary.pk_drug_sum_common_agent import (
    PKDrugSumCommonAgentResult,
)
from extractor.agents.pk_drug_summary.pk_drug_sum_workflow_utils import PKDrugSumWorkflowState

from extractor.agents.pk_drug_summary.pk_drug_sum_patient_info_refine_agent import (
    PatientInfoRefinedResult,
    get_patient_info_refine_prompt,
    post_process_refined_patient_info,
)


class PatientInfoRefinementStep(PKDrugSumCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Refining Population Information"
        self.end_title = "Completed to Refine Population Information"

    def get_system_prompt(self, state: PKDrugSumWorkflowState):
        title = state["title"]
        full_text = state["full_text"]
        md_table_drug = state["md_table_drug"]
        return get_patient_info_refine_prompt(title, full_text, md_table_drug)

    def leave_step(
        self,
        state: PKDrugSumWorkflowState,
        res: PKDrugSumCommonAgentResult,
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

    def get_post_processor_and_kwargs(self, state: PKDrugSumWorkflowState):
        md_table_drug = state["md_table_drug"]
        return post_process_refined_patient_info, {"md_table_drug": md_table_drug}
