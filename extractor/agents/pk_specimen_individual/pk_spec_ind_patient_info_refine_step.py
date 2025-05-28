from extractor.agents.pk_specimen_individual.pk_spec_ind_common_step import PKSpecIndCommonAgentStep
from extractor.agents.pk_specimen_individual.pk_spec_ind_common_agent import (
    PKSpecIndCommonAgentResult,
)
from extractor.agents.pk_specimen_individual.pk_spec_ind_workflow_utils import PKSpecIndWorkflowState

from extractor.agents.pk_specimen_individual.pk_spec_ind_patient_info_refine_agent import (
    PatientInfoRefinedResult,
    get_patient_info_refine_prompt,
    post_process_refined_patient_info,
)


class PatientInfoRefinementStep(PKSpecIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Refining Population Information"
        self.end_title = "Completed to Refine Population Information"

    def get_system_prompt(self, state: PKSpecIndWorkflowState):
        title = state["title"]
        full_text = state["full_text"]
        md_table_specimen = state["md_table_specimen"]
        return get_patient_info_refine_prompt(title, full_text, md_table_specimen)

    def leave_step(
        self,
        state: PKSpecIndWorkflowState,
        res: PKSpecIndCommonAgentResult,
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

    def get_post_processor_and_kwargs(self, state: PKSpecIndWorkflowState):
        md_table_specimen = state["md_table_specimen"]
        return post_process_refined_patient_info, {"md_table_specimen": md_table_specimen}
