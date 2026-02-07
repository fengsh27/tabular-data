from difflib import get_close_matches
import itertools
import pandas as pd
import re

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pk_population_individual.pk_popu_ind_common_step import PKPopuIndCommonStep


class RowCleanupStep(PKPopuIndCommonStep):
    """Row Cleanup"""

    def __init__(self):
        super().__init__()
        self.start_title = "Row Cleanup"
        self.end_title = "Completed Row Cleanup"

    def execute_directly(self, state):
        df_combined = state["df_combined"]
        df_combined["__original_order"] = range(len(df_combined))

        # delete if no num value
        df_combined = df_combined[df_combined['Main value'].str.contains(r'\d', na=False)]
        # def remove_sum_row(df):
        #     df = df.copy()
        #     try:
        #         df["Sample N"] = df["Sample N"].astype(int)
        #     except ValueError:
        #         return df
        #     total_sum = df["Sample N"].sum()
        #     for idx, value in df["Sample N"].items():
        #         if total_sum - value == value and 0 != value:
        #             df = df.drop(index=idx)
        #             break
        #
        #     return df.reset_index(drop=True)
        # df_combined = remove_sum_row(df_combined)
        #
        # def filter_max_sample_n(df):
        #     try:
        #         compare_cols = [col for col in df.columns if col not in ['Sample N', 'Population N', "Source text"]]
        #
        #         df_filtered = df.copy()
        #         df_filtered = df_filtered[df_filtered['Sample N'] == df_filtered['Sample N'].astype(int)]
        #
        #         df_result = df_filtered.sort_values('Sample N', ascending=False).drop_duplicates(subset=compare_cols,
        #                                                                                          keep='first')
        #         return df_result
        #     except ValueError:
        #         return df
        # df_combined = filter_max_sample_n(df_combined)
        #
        # # def expand_sample_time(df):
        # #     rows = []
        # #     for _, row in df.iterrows():
        # #         sample_time = str(row["Sample time"])
        # #         if "," in sample_time:
        # #             for time_val in sample_time.split(","):
        # #                 new_row = row.copy()
        # #                 new_row["Sample time"] = time_val.strip()
        # #                 rows.append(new_row)
        # #         else:
        # #             rows.append(row)
        # #     return pd.DataFrame(rows).reset_index(drop=True)
        # # df_combined = expand_sample_time(df_combined)
        #
        df_combined = df_combined.sort_values("__original_order").drop(columns="__original_order").reset_index(
            drop=True)

        return None, df_combined, {**DEFAULT_TOKEN_USAGE}

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["df_combined"] = processed_res
        return super().leave_step(state, res, processed_res, token_usage)
