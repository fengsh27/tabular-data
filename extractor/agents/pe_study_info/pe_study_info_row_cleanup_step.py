from difflib import get_close_matches
import itertools
import pandas as pd
import re

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pe_study_info.pe_study_info_common_step import PEStudyInfoCommonStep


class RowCleanupStep(PEStudyInfoCommonStep):
    """Row Cleanup"""

    def __init__(self):
        super().__init__()
        self.start_title = "Row Cleanup"
        self.end_title = "Completed Row Cleanup"

    def execute_directly(self, state):
        df_combined = state["df_combined"]

        new_order = ["Study type", "Population", "Study design", "Pregnancy stage", "Drug name", "Data source", "Inclusion criteria", "Exclusion criteria", "Outcomes", "Subject N"]
        df_combined = df_combined[new_order]

        df_combined["__original_order"] = range(len(df_combined))

        df_combined = df_combined.sort_values("__original_order").drop(columns="__original_order").reset_index(
            drop=True)

        return None, df_combined, {**DEFAULT_TOKEN_USAGE}

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["df_combined"] = processed_res
        return super().leave_step(state, res, processed_res, token_usage)
