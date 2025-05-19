import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pk_population_summary.pk_popu_sum_common_agent import PKPopuSumCommonAgentResult
from extractor.agents.pk_population_summary.pk_popu_sum_common_step import PKPopuSumCommonStep


class AssemblyStep(PKPopuSumCommonStep):
    """Assembly Step"""

    def __init__(self):
        super().__init__()
        self.start_title = "Assembly"
        self.end_title = "Completed Assembly"

    def execute_directly(self, state):
        md_table_characteristic = markdown_to_dataframe(state["md_table_characteristic"])
        df_table_patient_refined = markdown_to_dataframe(state["md_table_patient_refined"])
        md_table_characteristic_refined = markdown_to_dataframe(state["md_table_characteristic_refined"])

        assert (
            md_table_characteristic.shape[0]
            == df_table_patient_refined.shape[0]
            == md_table_characteristic_refined.shape[0]
        )

        df_characteristic = md_table_characteristic[["Population characteristic", "Characteristic sub-category", "Source text"]].copy()
        df_patient_filtered = df_table_patient_refined[
            ["Population", "Pregnancy stage", "Pediatric/Gestational age", "Population N"]].copy()
        df_characteristic_filtered = md_table_characteristic_refined[["Main value", "Unit", "Statistics type", "Variation type", "Variation value", "Interval type", "Lower bound", "Upper bound"]].copy()

        df_combined = pd.concat([df_characteristic, df_patient_filtered, df_characteristic_filtered], axis=1)
        new_order = [
            "Population characteristic",
            "Characteristic sub-category",
            "Unit",
            "Main value",
            "Statistics type",
            "Variation type",
            "Variation value",
            "Interval type",
            "Lower bound",
            "Upper bound",
            "Population",
            "Pregnancy stage",
            "Pediatric/Gestational age",
            "Population N",
            "Source text"
        ]
        df_combined = df_combined[new_order]
        self._step_output(
            state,
            step_output=f"""
Result:
{display_md_table(dataframe_to_markdown(df_combined))}
""",
        )

        return (
            PKPopuSumCommonAgentResult(reasoning_process=""),
            df_combined,
            {**DEFAULT_TOKEN_USAGE},
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["df_combined"] = processed_res
        return super().leave_step(state, res, processed_res, token_usage)
