from extractor.agents.pk_drug_individual.pk_drug_ind_common_step import PKDrugIndCommonAgentStep
from extractor.agents.pk_drug_individual.pk_drug_ind_common_agent import (
    PKDrugIndCommonAgentResult,
)
from extractor.agents.pk_drug_individual.pk_drug_ind_workflow_utils import PKDrugIndWorkflowState

from extractor.agents.pk_drug_individual.pk_drug_ind_drug_info_refine_agent import (
    DrugInfoRefinedResult,
    get_drug_info_refine_prompt,
    post_process_refined_drug_info,
)


class DrugInfoRefinementStep(PKDrugIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Refining Drug Information"
        self.end_title = "Completed to Refine Drug Information"

    def get_system_prompt(self, state: PKDrugIndWorkflowState):
        title = state["title"]
        full_text = state["full_text"]
        md_table_drug = state["md_table_drug"]
        system_prompt = get_drug_info_refine_prompt(title, full_text, md_table_drug)
        previous_errors_prompt = self._get_previous_errors_prompt(state)
        return system_prompt + previous_errors_prompt

    def leave_step(
        self,
        state: PKDrugIndWorkflowState,
        res: PKDrugIndCommonAgentResult,
        processed_res=None,
        token_usage=None,
    ):
        if processed_res is not None:
            state["md_table_drug_refined"] = processed_res
            self._step_output(state, step_output="Result (md_table_drug_refined):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, res, processed_res, token_usage)

    def get_schema(self):
        return DrugInfoRefinedResult

    def get_post_processor_and_kwargs(self, state: PKDrugIndWorkflowState):
        md_table_drug = state["md_table_drug"]
        return post_process_refined_drug_info, {"md_table_drug": md_table_drug}
