from difflib import get_close_matches
import itertools
import pandas as pd
import re

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pk_individual.pk_ind_common_step import PKIndCommonStep


class RowCleanupStep(PKIndCommonStep):
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
            "Patient ID",
            "Drug name",
            "Analyte",
            "Specimen",
            "Population",
            "Pregnancy stage",
            "Pediatric/Gestational age",
            "Parameter type",
            "Parameter unit",
            "Parameter value",
            "Time value",
            "Time unit"
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

        """Delete ERROR rows"""
        df_combined = df_combined[df_combined.ne("ERROR").all(axis=1)]
        """if Time unit == "weeks", Time value must be "N/A" ...... so it can be removed"""
        df_combined.loc[
            (df_combined["Time unit"] == "Weeks"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "weeks"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "Week"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "week"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "Wks"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "wks"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "Wk"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "wk"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "W"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "w"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "Months"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "months"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "Month"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "month"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "Years"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "years"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "Year"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "year"), "Time value"] = "N/A"

        """if Time == "N/A", Time unit must be "N/A" ......"""
        df_combined.loc[
            (df_combined["Time value"] == "N/A"), "Time unit"] = "N/A"
        df_combined.loc[
            (df_combined["Time unit"] == "N/A"), "Time value"] = "N/A"

        """if Value == "N/A", type and unit must be "N/A"。"""
        df_combined.loc[
            (df_combined["Parameter value"] == "N/A"), "Parameter type"] = "N/A"
        df_combined.loc[
            (df_combined["Parameter value"] == "N/A"), "Parameter unit"] = "N/A"

        """if Cmax, Cavg, Tmax, time value and unit must be "N/A"。"""
        df_combined.loc[
            (df_combined["Parameter type"] == "Cmax"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Parameter value"] == "Cmax"), "Time unit"] = "N/A"
        df_combined.loc[
            (df_combined["Parameter type"] == "Tmax"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Parameter value"] == "Tmax"), "Time unit"] = "N/A"
        df_combined.loc[
            (df_combined["Parameter type"] == "Cavg"), "Time value"] = "N/A"
        df_combined.loc[
            (df_combined["Parameter value"] == "Cavg"), "Time unit"] = "N/A"

        """replace empty by N/A"""
        df_combined.replace(r'^\s*$', 'N/A', regex=True, inplace=True)
        """replace n/a by N/A"""
        df_combined.replace("n/a", "N/A", inplace=True)
        """replace unknown by N/A"""
        df_combined.replace("unknown", "N/A", inplace=True)
        df_combined.replace("Unknown", "N/A", inplace=True)
        """replace nan by N/A"""
        df_combined.replace("nan", "N/A", inplace=True)
        """replace , by empty"""
        df_combined.replace(",", " ", inplace=True)

        """Remove N/A parameter value rows"""
        columns_to_check = ["Parameter value", ]

        df_combined = df_combined[
            ~df_combined[columns_to_check].apply(lambda row: any(str(cell) == 'N/A' for cell in row), axis=1)]
        df_combined = df_combined.reset_index(drop=True)

        """Remove duplicate"""
        df_combined = df_combined.drop_duplicates()
        df_combined = df_combined.reset_index(drop=True)

        df_combined["original_order"] = df_combined.index
        df_combined = (
            df_combined.sort_values(by="original_order")
            .drop(columns=["original_order"])
            .reset_index(drop=True)
        )

        """Patient ID as the first col"""
        cols = ['Patient ID'] + [col for col in df_combined.columns if col != 'Patient ID']
        df_combined = df_combined[cols]

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
