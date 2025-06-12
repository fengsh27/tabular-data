from extractor.agents.pe_study_info.pe_study_info_common_step import PEStudyInfoCommonAgentStep
from extractor.agents.pe_study_info.pe_study_info_common_agent import (
    PEStudyInfoCommonAgentResult,
)
from extractor.agents.pe_study_info.pe_study_info_design_info_agent import (
    STUDY_DESIGN_PROMPT,
    DesignInfoResult,
    post_process_study_design_info,
)
from extractor.agents.pe_study_info.pe_study_info_workflow_utils import PEStudyInfoWorkflowState


class DesignInfoExtractionStep(PEStudyInfoCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Extracting Study Design Information"
        self.end_title = "Completed to Extract Study Design Information"

    def get_system_prompt(self, state):
        title = state["title"]
        full_text = state["full_text"]
        return STUDY_DESIGN_PROMPT.format(
            title=title,
            full_text=full_text,
        )

    def leave_step(
        self,
        state: PEStudyInfoWorkflowState,
        res: PEStudyInfoCommonAgentResult,
        processed_res=None,
        token_usage=None,
    ):
        if processed_res is not None:
            state["md_table_design"] = processed_res
            self._step_output(state, step_output="Result (md_table_design):")
            self._step_output(state, step_output=processed_res)
        super().leave_step(state, res, processed_res, token_usage)

    def get_schema(self):
        return DesignInfoResult

    def get_post_processor_and_kwargs(self, state):
        return post_process_study_design_info, None
