from difflib import get_close_matches
import itertools
import pandas as pd
import re

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pe_study_outcome_ver2.pe_study_out_common_step import PEStudyOutCommonStep


class RowCleanupStep(PEStudyOutCommonStep):
    """Row Cleanup"""

    def __init__(self):
        super().__init__()
        self.start_title = "Row Cleanup"
        self.end_title = "Completed Row Cleanup"

    def execute_directly(self, state):
        df_combined = state["df_combined"]
        if df_combined.shape[0] == 0:  # empty table
            return None, df_combined, {**DEFAULT_TOKEN_USAGE}

        # df_combined["original_index"] = df_combined.index
        expected_columns = [
            "Characteristic",
            "Exposure",
            "Outcome",
            "Main value",
            "Main value unit",
            "Statistics type",
            "Variation type",
            "Variation value",
            "Interval type",
            "Lower bound",
            "Upper bound",
            "P value",
        ]

        def rename_columns(df, expected_columns):
            renamed_columns = {}
            for col in df.columns:
                matches = get_close_matches(col, expected_columns, n=1, cutoff=0.8)
                if matches:
                    renamed_columns[col] = matches[0]
                else:
                    renamed_columns[col] = col

            df.rename(columns=renamed_columns, inplace=True)
            return df

        df_combined = rename_columns(df_combined, expected_columns)

        """if Statistics type == Interval type or N/A, and (Main value == Lower bound or Main value == Upper bound), set Main value and Statistics type = N/A"""
        condition = (
            (df_combined["Statistics type"] == df_combined["Interval type"])
            | (df_combined["Statistics type"] == "N/A")
        ) & (
            (df_combined["Main value"] == df_combined["Lower bound"])
            | (df_combined["Main value"] == df_combined["Upper bound"])
        )
        df_combined.loc[condition, ["Main value", "Statistics type"]] = "N/A"
        """if Lower bound and Upper bound are both in Main value (string), Main value = N/A"""

        def contains_bounds(row):
            main_value = str(row["Main value"])
            lower = str(row["Lower bound"])
            upper = str(row["Upper bound"])
            if lower.strip() != "N/A" and upper.strip() != "N/A":
                return lower in main_value and upper in main_value
            return False

        mask = df_combined.apply(contains_bounds, axis=1)
        df_combined.loc[mask, "Main value"] = "N/A"
        """if Value == "N/A", Summary Statistics must be "N/A"。"""
        df_combined.loc[(df_combined["Main value"] == "N/A"), "Statistics type"] = "N/A"
        """if Lower limit & High limit == "N/A", Interval type must be "N/A"。"""
        df_combined.loc[
            (df_combined["Lower bound"] == "N/A")
            & (df_combined["Upper bound"] == "N/A"),
            "Interval type",
        ] = "N/A"
        """if Lower limit & High limit != "N/A", Interval type set as default "Range" """
        df_combined.loc[
            (df_combined["Lower bound"] != "N/A")
            & (df_combined["Upper bound"] != "N/A"),
            "Interval type",
        ] = "Range"
        """if Variation value == "N/A", Variation type must be "N/A"。"""
        df_combined.loc[(df_combined["Variation value"] == "N/A"), "Variation type"] = (
            "N/A"
        )
        """replace empty by N/A"""
        df_combined.replace(r"^\s*$", "N/A", regex=True, inplace=True)
        """replace n/a by N/A"""
        df_combined.replace("n/a", "N/A", inplace=True)
        """replace unknown by N/A"""
        df_combined.replace("unknown", "N/A", inplace=True)
        df_combined.replace("Unknown", "N/A", inplace=True)
        """replace nan by N/A"""
        df_combined.replace("nan", "N/A", inplace=True)
        """replace Standard Deviation (SD) by SD"""
        df_combined.replace("Standard Deviation (SD)", "SD", inplace=True)
        df_combined.replace("s.d.", "SD", inplace=True)
        df_combined.replace("S.D.", "SD", inplace=True)

        """replace , by empty"""
        df_combined.replace(",", " ", inplace=True)

        """Remove non-digit rows"""
        columns_to_check = [
            "Main value",
            "Variation type",
            "Lower bound",
            "Upper bound",
        ]

        def contains_number(s):
            return any(char.isdigit() for char in s)

        df_combined = df_combined[
            df_combined[columns_to_check].apply(
                lambda row: any(contains_number(str(cell)) for cell in row), axis=1
            )
        ]

        column_mapping = {
            "Main value": "Parameter value",
            "Statistics type": "Parameter statistic",
            "Main value unit": "Parameter unit",
        }
        df_combined = df_combined.rename(columns=column_mapping)

        new_order = [
            "Characteristic",
            "Exposure",
            "Outcome",
            "Parameter unit",
            "Parameter statistic",
            "Parameter value",
            "Variation type",
            "Variation value",
            "Interval type",
            "Lower bound",
            "Upper bound",
            "P value",
        ]
        df_combined = df_combined[new_order]

        df_combined["original_order"] = df_combined.index
        df_combined = (
            df_combined.sort_values(by="original_order")
            .drop(columns=["original_order"])
            .reset_index(drop=True)
        )

        self._step_output(
            state,
            step_output=f"""
{dataframe_to_markdown(df_combined)}
""",
        )

        return None, df_combined, {**DEFAULT_TOKEN_USAGE}

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            # update df_combined
            state["df_combined"] = processed_res
        return super().leave_step(state, res, processed_res, token_usage)
