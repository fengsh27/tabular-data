from TabFuncFlow.utils.table_utils import markdown_to_dataframe, dataframe_to_markdown
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pe_study_outcome.pe_study_out_common_agent import PEStudyOutCommonAgent
from extractor.agents.pe_study_outcome.pe_study_out_common_step import PEStudyOutCommonStep
from extractor.agents.pe_study_outcome.pe_study_out_characteristic_refine_agent import (
    get_characteristic_refine_prompt,
    CharacteristicRefinementResult,
    post_process_matched_list,
)
import pandas as pd


SUBTABLE_ROW_NUM = 16


class ParameterValueExtractionStep(PEStudyOutCommonStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Characteristic Refinement"
        self.end_title = "Completed Characteristic Refinement"

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
            system_prompt = get_characteristic_refine_prompt(md_table, md, caption)
            agent = PEStudyOutCommonAgent(llm=llm)
            res, processed_res, token_usage = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=INSTRUCTION_PROMPT,
                schema=CharacteristicRefinementResult,
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
            CharacteristicRefinementResult(reasoning_process="", extracted_param_values=[[]]),
            value_list,
            total_token_usage,
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            df_combined = state["df_combined"]
            df_param_value = pd.concat([markdown_to_dataframe(i) for i in processed_res], ignore_index=True)
            value_col_index = df_combined.columns.get_loc("Value")
            df_combined = df_combined.drop(columns=["Value"])
            for i, col in enumerate(df_param_value.columns):
                df_combined.insert(loc=value_col_index + i, column=col, value=df_param_value[col])

            state["df_combined"] = df_combined
            self._step_output(state, step_output="Result (df_combined):")
            self._step_output(state, step_output=str(processed_res))
        return super().leave_step(state, res, processed_res, token_usage)
