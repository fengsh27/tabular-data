from extractor.agents.pk_specimen_individual.pk_spec_ind_common_step import PKSpecIndCommonAgentStep
from extractor.agents.pk_specimen_individual.pk_spec_ind_common_agent import (
    PKSpecIndCommonAgentResult,
)
from extractor.agents.pk_specimen_individual.pk_spec_ind_specimen_info_agent import (
    SPECIMEN_INFO_PROMPT,
    SpecimenInfoResult,
    post_process_specimen_info,
)
from extractor.agents.pk_specimen_individual.pk_spec_ind_workflow_utils import PKSpecIndWorkflowState


class SpecimenInfoExtractionStep(PKSpecIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Extracting Specimen Information"
        self.end_title = "Completed to Extract Specimen Information"

    def get_system_prompt(self, state):
        title = state["title"]
        full_text = state["full_text"]
        return SPECIMEN_INFO_PROMPT.format(
            title=title,
            full_text=full_text,
        )

    def leave_step(
        self,
        state: PKSpecIndWorkflowState,
        res: PKSpecIndCommonAgentResult,
        processed_res=None,
        token_usage=None,
    ):
        if processed_res is not None:
            state["md_table_specimen"] = processed_res
            self._step_output(state, step_output="Result (md_table_specimen):")
            self._step_output(state, step_output=processed_res)
        super().leave_step(state, res, processed_res, token_usage)

    def get_schema(self):
        return SpecimenInfoResult

    def get_post_processor_and_kwargs(self, state):
        return post_process_specimen_info, None
