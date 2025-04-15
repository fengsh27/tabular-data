import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pk_summary.pk_sum_common_agent import PKSumCommonAgentResult
from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonStep


class AssemblyStep(PKSumCommonStep):
    """Assembly Step"""

    def __init__(self):
        super().__init__()
        self.start_title = "Assembly"
        self.end_title = "Completed Assembly"

    def execute_directly(self, state):
        drug_list = state["drug_list"]
        value_list = state["value_list"]
        patient_list = state["patient_list"]
        type_unit_list = state["type_unit_list"]

        df_list = []
        assert (
            len(drug_list)
            == len(patient_list)
            == len(type_unit_list)
            == len(value_list)
        )  # == len(time_list)
        for i in range(len(drug_list)):
            df_drug = markdown_to_dataframe(drug_list[i])
            df_table_patient = markdown_to_dataframe(patient_list[i])
            df_type_unit = markdown_to_dataframe(type_unit_list[i])
            df_value = markdown_to_dataframe(value_list[i])
            # df_time = markdown_to_dataframe(time_list[i])
            # df_combined = pd.concat([df_drug, df_table_patient, df_time, df_type_unit, df_value], axis=1)
            df_combined = pd.concat(
                [df_drug, df_table_patient, df_type_unit, df_value], axis=1
            )
            df_list.append(df_combined)
        if len(df_list) > 0:
            df_combined = pd.concat(df_list, ignore_index=True)
        else:
            df_combined = pd.DataFrame()

        self._step_output(
            state,
            step_output=f"""
Result:
{display_md_table(dataframe_to_markdown(df_combined))}
""",
        )

        return (
            PKSumCommonAgentResult(),
            df_combined,
            {**DEFAULT_TOKEN_USAGE},
            None,
        )

    def leave_step(self, state, step_reasoning_process=None, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["df_combined"] = processed_res
        return super().leave_step(state, step_reasoning_process, processed_res, token_usage)
