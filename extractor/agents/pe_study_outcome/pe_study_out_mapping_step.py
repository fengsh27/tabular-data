import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pe_study_outcome.pe_study_out_common_agent import PEStudyOutCommonAgentResult
from extractor.agents.pe_study_outcome.pe_study_out_common_step import PEStudyOutCommonStep

import logging
logger = logging.getLogger(__name__)

import re


class MappingStep(PEStudyOutCommonStep):
    """Mapping Step"""

    def __init__(self):
        super().__init__()
        self.start_title = "Mapping"
        self.end_title = "Completed Mapping"

    def execute_directly(self, state):
        col_mapping = state["col_mapping"]
        row_mapping = state["row_mapping"]
        md_table = state["md_table"]
        df_table = markdown_to_dataframe(md_table)
        columns = list(df_table.columns)
        records = []

        for row_idx, row in df_table.iterrows():
            for col in df_table.columns:
                if col_mapping.get(col) == "Row headers":
                    continue

                if col_mapping.get(col) == "P value":
                    continue

                left_col = None
                row_header = None
                for c in reversed(columns[:columns.index(col)]):
                    if col_mapping.get(c) == "Row headers":
                        left_col = c
                        row_header = row[c]
                        break

                if left_col is None:
                    better_row_header = None
                    # first_col = columns[0]
                    # better_row_header = row[first_col]
                else:
                    better_row_header = row_header
                    if row_header is not None:
                        for up_idx in range(row_idx - 1, -1, -1):
                            potential_subheader = df_table.loc[up_idx, left_col]
                            if row_mapping.get(left_col, {}).get(potential_subheader) == "Subheader":
                                better_row_header = f"{potential_subheader}; {row_header}"
                                break

                value = row[col]
                col_type = col_mapping.get(col)
                row_type = None
                if better_row_header and left_col in row_mapping:
                    row_type = row_mapping[left_col].get(row_header)

                logger.info(
                    f"[{value} | "
                    f"Row {better_row_header} (type: {row_type}) | "
                    f"Col {col} (type: {col_type})]"
                )

                if col_type == "Subheader" or row_type == "Subheader":
                    continue

                p_value = None
                col_idx = columns.index(col)
                for next_col in columns[col_idx + 1:]:
                    if col_mapping.get(next_col) == "P value":
                        p_value = row[next_col]
                        break

                record = {
                    "Characteristic": "N/A",
                    "Exposure": "N/A",
                    "Outcome": "N/A",
                    "Value": value,
                    "P value": p_value
                }

                if row_type == "Characteristic":
                    record["Characteristic"] = better_row_header
                elif row_type == "Exposure":
                    record["Exposure"] = better_row_header
                elif row_type == "Outcome":
                    record["Outcome"] = better_row_header

                if col_type == "Characteristic":
                    record["Characteristic"] = col
                elif col_type == "Exposure":
                    record["Exposure"] = col
                elif col_type == "Outcome":
                    record["Outcome"] = col

                if row_type == col_type == "Characteristic":
                    record["Characteristic"] = better_row_header + "; " + col
                elif row_type == col_type == "Exposure":
                    record["Exposure"] = better_row_header + "; " + col
                elif row_type == col_type == "Outcome":
                    record["Outcome"] = better_row_header + "; " + col

                records.append(record)

        df_combined = pd.DataFrame(records)

        def has_number(s):
            return bool(re.search(r'\d', str(s)))

        df_combined = df_combined[
            df_combined["Value"].apply(has_number)  # | df_combined["P value"].apply(has_number)
            ].reset_index(drop=True)

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
