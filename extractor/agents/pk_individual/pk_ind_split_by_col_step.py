from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE
from extractor.agents.pk_individual.pk_ind_common_step import PKIndCommonAgentStep
from extractor.agents.pk_individual.pk_ind_split_by_col_agent import (
    get_split_by_columns_prompt,
    SplitByColumnsResult,
    post_process_split_by_columns,
)


class SplitByColumnsStep(PKIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Sub-table Creation"
        self.end_title = "Completed Sub-table Creation"

    def get_system_prompt(self, state):
        md_table_aligned = state["md_table_aligned"]
        col_mapping = state["col_mapping"]
        system_prompt = get_split_by_columns_prompt(md_table_aligned, col_mapping)
        previous_errors_prompt = self._get_previous_errors_prompt(state)
        return system_prompt + previous_errors_prompt

    def get_schema(self):
        return SplitByColumnsResult

    def get_post_processor_and_kwargs(self, state):
        md_table_aligned = state["md_table_aligned"]
        return post_process_split_by_columns, {"md_table_aligned": md_table_aligned}

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        col_mapping = state["col_mapping"]
        tmp_md_table_list = []
        md_table_list = processed_res
        for md in md_table_list:
            df = markdown_to_dataframe(md)
            cols_to_drop = [
                col for col in df.columns if col_mapping.get(col) == "Uncategorized"
            ]
            df.drop(columns=cols_to_drop, inplace=True)
            tmp_md_table_list.append(dataframe_to_markdown(df))
        # md_table_list = _md_table_list
        tmp_md_table_list_1 = []
        for md in tmp_md_table_list:
            df = markdown_to_dataframe(md)
            cols_to_split = [
                col for col in df.columns if col_mapping.get(col) == "Parameter value"
            ]
            common_cols = [col for col in df.columns if col not in cols_to_split]
            for col in cols_to_split:
                if col in df.columns:
                    selected_cols = [
                        c for c in df.columns if c in common_cols or c == col
                    ]
                    tmp_md_table_list_1.append(
                        dataframe_to_markdown(df[selected_cols].copy())
                    )
        md_table_list = tmp_md_table_list_1
        tmp_md_table_list_2 = []
        for md in md_table_list:
            df = markdown_to_dataframe(md)
            parameter_col = [col for col in df.columns if col_mapping.get(col) == "Parameter value"][0]
            patient_col = [col for col in df.columns if col_mapping.get(col) == "Patient ID"][0]
            df_reshaped = df.rename(columns={patient_col: patient_col, parameter_col: "Parameter value"})
            df_reshaped["Parameter type"] = parameter_col
            df_reshaped = df_reshaped[[patient_col, "Parameter type", "Parameter value"]]
            tmp_md_table_list_2.append(dataframe_to_markdown(df_reshaped))
        md_table_list = tmp_md_table_list_2
        col_mapping["Parameter type"] = "Parameter type"
        # Note: Ensure 'Parameter type' is present in col_mapping for type-unit extraction.
        # Note: This is also handled in 'type_unit_extract_step' — this serves as a double-check for robustness.
        state["md_table_list"] = md_table_list
        self._step_output(state, step_output="Result (md_table_list):")
        self._step_output(state, step_output=str(md_table_list))
        return super().leave_step(state, res, md_table_list, token_usage)

    # override super().execute_directly
    def execute_directly(self, state):
        col_mapping = state["col_mapping"]
        md_table_aligned = state["md_table_aligned"]
        parameter_count = list(col_mapping.values()).count("Parameter value")
        patient_id_count = list(col_mapping.values()).count("Patient ID")

        if parameter_count > 1 or patient_id_count > 1:
            return super().execute_directly(state)
        else:
            res = SplitByColumnsResult(
                reasoning_process="",
                sub_tables_columns=[[]],
            )
            processed_res = [
                md_table_aligned,
            ]
            return res, processed_res, {**DEFAULT_TOKEN_USAGE}
