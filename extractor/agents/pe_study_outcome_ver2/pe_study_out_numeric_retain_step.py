import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pe_study_outcome_ver2.pe_study_out_common_agent import PEStudyOutCommonAgentResult
from extractor.agents.pe_study_outcome_ver2.pe_study_out_common_step import PEStudyOutCommonStep

import logging
logger = logging.getLogger(__name__)

import re


class NumericRetainStep(PEStudyOutCommonStep):
    """Numeric Retain Step"""

    def __init__(self):
        super().__init__()
        self.start_title = "Numeric Retain"
        self.end_title = "Completed Numeric Retain"

    def execute_directly(self, state):
        md_table = state["md_table"]
        df_table = markdown_to_dataframe(md_table)
        records = []

        def has_number(s):
            return bool(re.search(r'\d', str(s)))

        for row_idx, row in df_table.iterrows():
            for col in df_table.columns:
                value = row[col]
                if has_number(value):
                    records.append({"Value": value})

        df_combined = pd.DataFrame(records)

        return (
            PEStudyOutCommonAgentResult(reasoning_process=""),
            df_combined,
            {**DEFAULT_TOKEN_USAGE},
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["df_combined"] = processed_res
            self._step_output(state, step_output=dataframe_to_markdown(processed_res))
        return super().leave_step(state, res, processed_res, token_usage)
