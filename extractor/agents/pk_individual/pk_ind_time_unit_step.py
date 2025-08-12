import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_individual.pk_ind_time_unit_agent import (
    get_time_and_unit_prompt,
    TimeAndUnitResult,
    post_process_time_and_unit,
)

from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_individual.pk_ind_common_step import PKIndCommonStep
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgent


class TimeExtractionStep(PKIndCommonStep):
    "Time Extraction Step"

    def __init__(self):
        super().__init__()
        self.start_title = "Time Extraction"
        self.end_title = "Completed Time Extraction"

    def execute_directly(self, state):
        time_list = []
        md_table_list = state["md_table_list"]
        md_table_aligned = state["md_table_aligned"]
        llm = state["llm"]
        caption = state["caption"]
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        round = 0
        for md in md_table_list:
            round += 1
            self._step_output(f"Trial {round}")
            system_prompt = get_time_and_unit_prompt(md_table_aligned, md, caption)
            previous_errors_prompt = self._get_previous_errors_prompt(state)
            system_prompt = system_prompt + previous_errors_prompt
            instruction_prompt = INSTRUCTION_PROMPT
            agent = PKIndCommonAgent(llm=llm)
            res, processed_res, token_usage = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=TimeAndUnitResult,
                post_process=post_process_time_and_unit,
                md_table_post_processed=md,
            )
            self._step_output(
                state,
                step_reasoning_process=res.reasoning_process if res is not None else "",
            )
            time_append_list: list[list[str]] = processed_res
            time_list.append(time_append_list)
            total_token_usage = increase_token_usage(total_token_usage, token_usage)

        return (
            TimeAndUnitResult(
                reasoning_process="",
                times_and_units=[]
            ),
            time_list,
            total_token_usage,
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["time_list"] = processed_res
            self._step_output(state, step_output="Result (time_list):")
            self._step_output(state, step_output=str(processed_res))
        return super().leave_step(state, res, processed_res, token_usage)
