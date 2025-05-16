import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pk_drug_summary.pk_drug_sum_common_agent import PKDrugSumCommonAgentResult
from extractor.agents.pk_drug_summary.pk_drug_sum_common_step import PKDrugSumCommonStep


class AssemblyStep(PKDrugSumCommonStep):
    """Assembly Step"""

    def __init__(self):
        super().__init__()
        self.start_title = "Assembly"
        self.end_title = "Completed Assembly"

    def execute_directly(self, state):
        df_table_drug = markdown_to_dataframe(state["md_table_drug"])
        df_table_patient_refined = markdown_to_dataframe(state["md_table_patient_refined"])
        # df_table_time = markdown_to_dataframe(state["md_table_time"])
        df_table_drug_refined = markdown_to_dataframe(state["md_table_drug_refined"])

        assert (
            df_table_drug.shape[0]
            == df_table_patient_refined.shape[0]
            == df_table_drug_refined.shape[0]
        )

        df_note = df_table_drug[["Source text"]].copy()
        df_patient_filtered = df_table_patient_refined[
            ["Population", "Pregnancy stage", "Pediatric/Gestational age", "Population N"]].copy()
        df_drug_filtered = df_table_drug_refined[[
            "Drug/Metabolite name",
            "Dose amount",
            "Dose unit",
            "Dose frequency",
            "Dose schedule",
            "Dose route"
        ]].copy()

        # combined_source = df_specimen_filtered["Source text"].fillna('') + '\n' + \
        #                   df_patient_filtered["Source text"].fillna('') + '\n' + \
        #                   df_time_filtered["Source text"].fillna('')
        #
        # df_specimen_filtered.drop(columns=["Source text"], inplace=True)
        # df_patient_filtered.drop(columns=["Source text"], inplace=True)
        # df_time_filtered.drop(columns=["Source text"], inplace=True)

        df_combined = pd.concat([df_drug_filtered, df_patient_filtered, df_note], axis=1)

        # df_combined["Source text"] = combined_source

        self._step_output(
            state,
            step_output=f"""
Result:
{display_md_table(dataframe_to_markdown(df_combined))}
""",
        )

        return (
            PKDrugSumCommonAgentResult(reasoning_process=""),
            df_combined,
            {**DEFAULT_TOKEN_USAGE},
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["df_combined"] = processed_res
        return super().leave_step(state, res, processed_res, token_usage)
