import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pe_study_info.pe_study_info_common_agent import PEStudyInfoCommonAgentResult
from extractor.agents.pe_study_info.pe_study_info_common_step import PEStudyInfoCommonStep


class AssemblyStep(PEStudyInfoCommonStep):
    """Assembly Step"""

    def __init__(self):
        super().__init__()
        self.start_title = "Assembly"
        self.end_title = "Completed Assembly"

    def execute_directly(self, state):
        df_table_design = markdown_to_dataframe(state["md_table_design"])
        df_table_design_refined = markdown_to_dataframe(state["md_table_design_refined"])

        df_combined = pd.concat([df_table_design_refined, df_table_design], axis=1)

        self._step_output(
            state,
            step_output=f"""
{dataframe_to_markdown(df_combined)}
""",
        )

        return (
            PEStudyInfoCommonAgentResult(reasoning_process=""),
            df_combined,
            {**DEFAULT_TOKEN_USAGE},
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["df_combined"] = processed_res
        return super().leave_step(state, res, processed_res, token_usage)
