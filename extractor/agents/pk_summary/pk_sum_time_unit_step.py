import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_summary.pk_sum_time_unit_agent import (
    get_time_and_unit_prompt,
    TimeAndUnitResult,
    post_process_time_and_unit,
)
from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonAgentStep


class TimeExtractionStep(PKSumCommonAgentStep):
    "Time Extraction Step"

    def __init__(self):
        super().__init__()
        self.start_title = "Time Extraction"
        self.end_title = "Completed Time Extraction"

    def get_system_prompt(self, state):
        md_table_aligned = state["md_table_aligned"]
        caption = state["caption"]
        df_combined = state["df_combined"]
        if df_combined.shape[0] == 0:
            self.md_data_lines_after_post_process = dataframe_to_markdown(df_combined)
        else:
            self.md_data_lines_after_post_process = dataframe_to_markdown(
                df_combined[
                    [
                        "Main value",
                        "Statistics type",
                        "Variation type",
                        "Variation value",
                        "Interval type",
                        "Lower bound",
                        "Upper bound",
                        "P value",
                    ]
                ]
            )
        system_prompt = get_time_and_unit_prompt(
            md_table_aligned=md_table_aligned,
            md_table_post_processed=self.md_data_lines_after_post_process,
            caption=caption,
        )
        previous_errors_prompt = self._get_previous_errors_prompt(state)
        return system_prompt + previous_errors_prompt

    def get_schema(self):
        return TimeAndUnitResult

    def get_post_processor_and_kwargs(self, state):
        return post_process_time_and_unit, {
            "md_table_post_processed": self.md_data_lines_after_post_process
        }

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is None:
            return super().leave_step(state, res, processed_res, token_usage)
        md_table_time: str = processed_res
        df_combined = state["df_combined"]
        df_combined = pd.concat(
            [df_combined, markdown_to_dataframe(md_table_time)], axis=1
        )
        df_combined = df_combined.reset_index(drop=True)
        self._step_output(state, step_output="Result (df_combined):")
        self._step_output(
            state, step_output=display_md_table(dataframe_to_markdown(df_combined))
        )

        # update state['df_combined]
        state["df_combined"] = df_combined
        super().leave_step(state, res, processed_res, token_usage)
