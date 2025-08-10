from extractor.agents.pk_drug_individual.pk_drug_ind_common_step import PKDrugIndCommonAgentStep
from extractor.agents.pk_drug_individual.pk_drug_ind_common_agent import (
    PKDrugIndCommonAgentResult,
)
from extractor.agents.pk_drug_individual.pk_drug_ind_drug_info_agent import (
    DRUG_INFO_PROMPT,
    DrugInfoResult,
    post_process_population_info,
)
from extractor.agents.pk_drug_individual.pk_drug_ind_workflow_utils import PKDrugIndWorkflowState


class DrugInfoExtractionStep(PKDrugIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Extracting Drug Information"
        self.end_title = "Completed to Extract Drug Information"

    def get_system_prompt(self, state):
        title = state["title"]
        full_text = state["full_text"]
        system_prompt = DRUG_INFO_PROMPT.format(
            title=title,
            full_text=full_text,
        )
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
            state["md_table_drug"] = processed_res
            self._step_output(state, step_output="Result (md_table_drug):")
            self._step_output(state, step_output=processed_res)
        super().leave_step(state, res, processed_res, token_usage)

    def get_schema(self):
        return DrugInfoResult

    def get_post_processor_and_kwargs(self, state):
        return post_process_population_info, None
