from TabFuncFlow.utils.table_utils import markdown_to_dataframe, dataframe_to_markdown
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pe_study_outcome_ver2.pe_study_out_common_agent import PEStudyOutCommonAgent
from extractor.agents.pe_study_outcome_ver2.pe_study_out_common_step import PEStudyOutCommonStep
from extractor.agents.pe_study_outcome_ver2.pe_study_out_param_value_agent import (
    get_parameter_value_prompt,
    ParameterValueResult,
    post_process_matched_list,
)
import pandas as pd


SUBTABLE_ROW_NUM = 10


class ParameterValueExtractionStep(PEStudyOutCommonStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Parameter Value Extraction"
        self.end_title = "Completed Parameter Value Extraction"

    def execute_directly(self, state):
        md_table = state["md_table"]
        llm = state["llm"]
        df_combined = state["df_combined"]
        caption = state["caption"]
        md_table_list = [dataframe_to_markdown(df_combined.iloc[i:i + SUBTABLE_ROW_NUM]) for i in range(0, len(df_combined), SUBTABLE_ROW_NUM)]

        value_list = []
        round = 0
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        for md in md_table_list:
            round += 1
            self._step_output(f"Trial {round}")
            system_prompt = get_parameter_value_prompt(md_table, md, caption)
            previous_errors_prompt = self._get_previous_errors_prompt(state)
            system_prompt = system_prompt + previous_errors_prompt
            agent = PEStudyOutCommonAgent(llm=llm)
            res, processed_res, token_usage = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=INSTRUCTION_PROMPT,
                schema=ParameterValueResult,
                post_process=post_process_matched_list,
                expected_rows=markdown_to_dataframe(md).shape[0],
            )
            self._step_output(
                state,
                step_reasoning_process=res.reasoning_process if res is not None else "",
            )
            value_list.append(processed_res)
            total_token_usage = increase_token_usage(token_usage)

        return (
            ParameterValueResult(reasoning_process="", extracted_param_values=[[]]),
            value_list,
            total_token_usage,
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:

            df_param_value = pd.concat([markdown_to_dataframe(i) for i in processed_res], ignore_index=True)
            other_columns = [col for col in df_param_value.columns if col != "P value"]
            mask = df_param_value[other_columns].applymap(pd.to_numeric, errors='coerce').notna().any(axis=1)
            state["df_combined"] = df_param_value[mask].copy()

            self._step_output(state, step_output="Result (df_combined):")
            self._step_output(state, step_output=str(processed_res))

        return super().leave_step(state, res, processed_res, token_usage)

