from extractor.agents.pe_study_info.pe_study_info_common_step import PEStudyInfoCommonAgentStep
from extractor.agents.pe_study_info.pe_study_info_common_agent import (
    PEStudyInfoCommonAgentResult,
)
from extractor.agents.pe_study_info.pe_study_info_workflow_utils import PEStudyInfoWorkflowState

from extractor.agents.pe_study_info.pe_study_info_design_info_refine_agent import (
    DesignInfoRefinedResult,
    get_design_info_refine_prompt,
    post_process_refined_design_info,
)


class DesignInfoRefinementStep(PEStudyInfoCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Refining Design Information"
        self.end_title = "Completed to Refine Design Information"

    def get_system_prompt(self, state: PEStudyInfoWorkflowState):
        title = state["title"]
        full_text = state["full_text"]
        md_table_design = state["md_table_design"]
        return get_design_info_refine_prompt(title, full_text, md_table_design)

    def leave_step(
        self,
        state: PEStudyInfoWorkflowState,
        res: PEStudyInfoCommonAgentResult,
        processed_res=None,
        token_usage=None,
    ):
        if processed_res is not None:
            state["md_table_design_refined"] = processed_res
            self._step_output(state, step_output="Result (md_table_design_refined):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, res, processed_res, token_usage)

    def get_schema(self):
        return DesignInfoRefinedResult

    def get_post_processor_and_kwargs(self, state: PEStudyInfoWorkflowState):
        md_table_design = state["md_table_design"]
        return post_process_refined_design_info, {"md_table_design": md_table_design}
