import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pk_specimen_summary.pk_spec_sum_common_agent import PKSpecSumCommonAgentResult
from extractor.agents.pk_specimen_summary.pk_spec_sum_common_step import PKSpecSumCommonStep


class AssemblyStep(PKSpecSumCommonStep):
    """Assembly Step"""

    def __init__(self):
        super().__init__()
        self.start_title = "Assembly"
        self.end_title = "Completed Assembly"

    def execute_directly(self, state):
        df_table_specimen = markdown_to_dataframe(state["md_table_specimen"])
        df_table_patient_refined = markdown_to_dataframe(state["md_table_patient_refined"])
        df_table_time = markdown_to_dataframe(state["md_table_time"])

        assert (
            df_table_specimen.shape[0]
            == df_table_patient_refined.shape[0]
            == df_table_time.shape[0]
        )

        df_specimen_filtered = df_table_specimen[["Specimen", "Sample N"]].copy()
        df_patient_filtered = df_table_patient_refined[
            ["Population", "Pregnancy stage", "Pediatric/Gestational age", "Population N"]].copy()
        df_time_filtered = df_table_time[["Sample time", "Time unit", "Source text"]].copy()

        # combined_source = df_specimen_filtered["Source text"].fillna('') + '\n' + \
        #                   df_patient_filtered["Source text"].fillna('') + '\n' + \
        #                   df_time_filtered["Source text"].fillna('')
        #
        # df_specimen_filtered.drop(columns=["Source text"], inplace=True)
        # df_patient_filtered.drop(columns=["Source text"], inplace=True)
        # df_time_filtered.drop(columns=["Source text"], inplace=True)

        df_combined = pd.concat([df_specimen_filtered, df_patient_filtered, df_time_filtered], axis=1)

        # df_combined["Source text"] = combined_source

        self._step_output(
            state,
            step_output=f"""
Result:
{display_md_table(dataframe_to_markdown(df_combined))}
""",
        )

        return (
            PKSpecSumCommonAgentResult(reasoning_process=""),
            df_combined,
            {**DEFAULT_TOKEN_USAGE},
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["df_combined"] = processed_res
        return super().leave_step(state, res, processed_res, token_usage)
