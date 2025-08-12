from typing import List, Tuple
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgent
from extractor.agents.pk_individual.pk_ind_common_step import PKIndCommonAgentStep
from extractor.agents.pk_individual.pk_ind_param_type_unit_extract_agent import (
    ExtractedParamTypeUnits,
    get_param_type_unit_extraction_prompt,
    ParamTypeUnitExtractionResult,
    post_process_validate_matched_tuple,
)


class ExtractParamTypeAndUnitStep(PKIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Parameter Type, Unit and Value Extraction"
        self.end_title = "Completed Parameter Type, Unit and Value Extraction"

    def get_system_prompt(self, state):
        """get system prompt, we won't use this method in this step"""
        return ""

    def get_schema(self):
        return ParamTypeUnitExtractionResult

    def get_post_processor_and_kwargs(self, state):
        """get post-processor and its kwargs, we won't use this method in this step"""
        return post_process_validate_matched_tuple, None

    def execute_directly(self, state):
        md_table_list = state["md_table_list"]
        col_mapping = state["col_mapping"]
        col_mapping["Parameter type"] = "Parameter type"
        # Note: Ensure 'Parameter type' is present in col_mapping for type-unit extraction.
        # Note: This is also handled in 'split_by_col_step' — this serves as a double-check for robustness.
        type_unit_list: list[str] = []

        round = 0
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        for md in md_table_list:
            # df = markdown_to_dataframe(md)
            round += 1
            step_name = f" (Trial {str(round)})"
            self._step_output(state, step_output=step_name)
            llm = state["llm"]
            md_table_aligned = state["md_table_aligned"]
            caption = state["caption"]
            schema = self.get_schema()
            system_prompt = get_param_type_unit_extraction_prompt(
                md_table_aligned, md, col_mapping, caption
            )
            previous_errors_prompt = self._get_previous_errors_prompt(state)
            system_prompt = system_prompt + previous_errors_prompt
            instruction_prompt = self.get_instruction_prompt(state)
            agent = PKIndCommonAgent(llm=llm)
            res, processed_res, token_usage = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
                post_process=post_process_validate_matched_tuple,
                md_table=md,
                col_mapping=col_mapping,
            )
            self._step_output(
                state,
                step_reasoning_process=res.reasoning_process
                if res is not None
                else "",
            )
            tuple_type_unit: Tuple[List[str], List[str], List[str]] = processed_res

            md_type_unit = dataframe_to_markdown(
                pd.DataFrame(
                    [tuple_type_unit[0], tuple_type_unit[1], tuple_type_unit[2]],
                    index=["Parameter type", "Parameter unit", "Parameter value"],
                ).T
            )
            type_unit_list.append(md_type_unit)
            total_token_usage = increase_token_usage(
                total_token_usage, token_usage
            )

        return (
            ParamTypeUnitExtractionResult(
                reasoning_process="",
                extracted_param_units=ExtractedParamTypeUnits(
                    parameter_types=[], parameter_units=[], parameter_values=[]
                ),
            ),
            type_unit_list,
            total_token_usage,
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["type_unit_list"] = processed_res
            self._step_output(state, step_output="Result (type_unit_list):")
            self._step_output(state, step_output=str(processed_res))
        return super().leave_step(state, res, processed_res, token_usage)
