from extractor.agents.agent_utils import (
    display_md_table,
)
from extractor.agents.pk_individual.pk_ind_common_step import PKIndCommonAgentStep
from extractor.agents.pk_individual.pk_ind_common_agent import (
    PKIndCommonAgentResult,
)
from extractor.agents.pk_individual.pk_ind_drug_info_agent import (
    DRUG_INFO_PROMPT,
    DrugInfoResult,
    post_process_drug_info,
)
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState


class DrugInfoExtractionStep(PKIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Extracting Drug Information"
        self.end_title = "Completed to Extract Drug Information"

    def get_system_prompt(self, state):
        md_table = state["md_table"]
        caption = state["caption"]
        title = state["title"]
        title = title if title is not None else "N/A"
        return DRUG_INFO_PROMPT.format(
            processed_md_table=display_md_table(md_table),
            caption=caption,
            paper_title=title,
        )

    def leave_step(
        self,
        state: PKIndWorkflowState,
        res: PKIndCommonAgentResult,
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
        return post_process_drug_info, None
