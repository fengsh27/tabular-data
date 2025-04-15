from difflib import get_close_matches
import itertools
import pandas as pd
import re

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonStep


class RowCleanupStep(PKSumCommonStep):
    """Row Cleanup"""

    def __init__(self):
        super().__init__()
        self.start_title = "Row Cleanup"
        self.end_title = "Completed Row Cleanup"

    def execute_directly(self, state):
        df_combined = state["df_combined"]
        if df_combined.shape[0] == 0:  # empty table
            return None, df_combined, {**DEFAULT_TOKEN_USAGE}, None

        # df_combined["original_index"] = df_combined.index
        expected_columns = [
            "Drug name",
            "Analyte",
            "Specimen",
            "Population",
            "Pregnancy stage",
            "Pediatric/Gestational age",
            "Subject N",
            "Parameter type",
            "Parameter unit",
            "Main value",
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

        """Delete ERROR rows"""
        df_combined = df_combined[df_combined.ne("ERROR").all(axis=1)]
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

        """ Merge """

        df = df_combined.copy()

        df.replace("N/A", pd.NA, inplace=True)

        # group_columns = ["Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage", "Subject N", "Parameter type",
        #                  "Parameter unit"]
        group_columns = [
            "Drug name",
            "Analyte",
            "Specimen",
            "Population",
            "Pregnancy stage",
            "Pediatric/Gestational age",
            "Subject N",
            "Parameter type",
            "Parameter unit",
        ]
        grouped = df.groupby(group_columns, dropna=False)

        merged_rows = []
        for _, group in grouped:
            used_indices = set()

            for i, j in itertools.combinations(range(len(group)), 2):
                if i in used_indices or j in used_indices:
                    continue

                row1, row2 = group.iloc[i].copy(), group.iloc[j].copy()
                can_merge = True

                for col in df.columns:
                    val1, val2 = row1[col], row2[col]
                    if pd.notna(val1) and pd.notna(val2) and val1 != val2:
                        can_merge = False
                        break
                    elif pd.isna(val1) and pd.notna(val2):
                        row1[col] = val2
                    elif pd.notna(val1) and pd.isna(val2):
                        row2[col] = val1

                if can_merge:
                    used_indices.add(i)
                    used_indices.add(j)
                    merged_rows.append(row1)
                # else:
                #     merged_rows.append(row1)
                #     merged_rows.append(row2)

            for i in range(len(group)):
                if i not in used_indices:
                    merged_rows.append(group.iloc[i])

        df_merged = pd.DataFrame(merged_rows, columns=df.columns)
        df_merged.fillna("N/A", inplace=True)

        df_combined = df_merged

        """Remove duplicate"""
        df_combined = df_combined.drop_duplicates()

        """delete 'fill in subject N as value error', this implementation is bad, still looking for better solutions"""
        df_combined = df_combined[df_combined["Subject N"] != df_combined["Main value"]]

        """fix put range only in lower limit/high limit"""
        float_pattern = re.compile(r"-?\d+\.\d+")

        def extract_limits(row):
            if row["Upper bound"] == "N/A":
                numbers = float_pattern.findall(str(row["Lower bound"]))
                if len(numbers) == 2:
                    return pd.Series([str(numbers[0]), str(numbers[1])])
            if row["Lower bound"] == "N/A":
                numbers = float_pattern.findall(str(row["Upper bound"]))
                if len(numbers) == 2:
                    return pd.Series([str(numbers[0]), str(numbers[1])])
            if row["Upper bound"] == row["Lower bound"]:
                numbers = float_pattern.findall(str(row["Upper bound"]))
                if len(numbers) == 2:
                    return pd.Series([str(numbers[0]), str(numbers[1])])
            return pd.Series([row["Lower bound"], row["Upper bound"]])

        df_combined[["Lower bound", "Upper bound"]] = df_combined.apply(
            extract_limits, axis=1
        )

        """remove inclusive rows"""

        def remove_contained_rows(df):
            df_cleaned = df.copy()

            rows_to_drop = set()
            for i in range(len(df_cleaned)):
                for j in range(i + 1, len(df_cleaned)):
                    row1 = df_cleaned.iloc[i]
                    row2 = df_cleaned.iloc[j]

                    if all((r1 == r2) or (r1 == "N/A") for r1, r2 in zip(row1, row2)):
                        rows_to_drop.add(i)  # row1 included by row2
                    elif all((r2 == r1) or (r2 == "N/A") for r1, r2 in zip(row1, row2)):
                        rows_to_drop.add(j)

            df_cleaned = df_cleaned.drop(index=rows_to_drop)
            return df_cleaned

        df_combined = remove_contained_rows(df_combined)
        df_combined = remove_contained_rows(df_combined)
        df_combined = remove_contained_rows(df_combined)
        df_combined = remove_contained_rows(df_combined)
        df_combined = remove_contained_rows(df_combined)

        """col exchange"""
        cols = list(df_combined.columns)
        i, j = cols.index("Main value"), cols.index("Statistics type")
        cols[i], cols[j] = cols[j], cols[i]
        df_combined = df_combined[cols]

        """give range to median"""
        # group_columns = ["Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage", "Subject N", "Parameter type",
        #                  "Parameter unit"]
        group_columns = [
            "Drug name",
            "Analyte",
            "Specimen",
            "Population",
            "Pregnancy stage",
            "Pediatric/Gestational age",
            "Subject N",
            "Parameter type",
            "Parameter unit",
        ]

        # Finding pairs of rows that match on group_columns
        grouped = df_combined.groupby(group_columns)

        # Processing each group
        for _, group in grouped:
            if (
                len(group) == 2
            ):  # Only process if there are exactly two rows in the group
                median_row = group[group["Statistics type"] == "Median"]
                non_median_row = group[group["Statistics type"] != "Median"]

                if not median_row.empty and not non_median_row.empty:
                    # Check if non-median row has "Range"
                    if "Range" in non_median_row["Interval type"].values:
                        # Assign range values to the median row
                        df_combined.loc[
                            median_row.index,
                            ["Interval type", "Lower bound", "Upper bound"],
                        ] = non_median_row[
                            ["Interval type", "Lower bound", "Upper bound"]
                        ].values

                        # Remove range information from the non-median row
                        df_combined.loc[
                            non_median_row.index,
                            ["Interval type", "Lower bound", "Upper bound"],
                        ] = ["N/A", "N/A", "N/A"]

        df_combined["original_order"] = df_combined.index
        df_combined = (
            df_combined.sort_values(by="original_order")
            .drop(columns=["original_order"])
            .reset_index(drop=True)
        )

        self._step_output(
            state,
            step_output=f"""
Result:
{display_md_table(dataframe_to_markdown(df_combined))}
""",
        )

        return None, df_combined, {**DEFAULT_TOKEN_USAGE}, None

    def leave_step(self, state, step_reasoning_process, processed_res=None, token_usage=None):
        if processed_res is not None:
            # update df_combined
            state["df_combined"] = processed_res
        return super().leave_step(state, step_reasoning_process, processed_res, token_usage)
