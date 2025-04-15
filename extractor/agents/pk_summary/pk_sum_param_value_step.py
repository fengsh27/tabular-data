from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_summary.pk_sum_common_agent import PKSumCommonAgent
from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonStep
from extractor.agents.pk_summary.pk_sum_param_value_agent import (
    get_parameter_value_prompt,
    ParameterValueResult,
    post_process_matched_list,
)


class ParameterValueExtractionStep(PKSumCommonStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Parameter Value Extraction"
        self.end_title = "Completed Parameter Value Extraction"

    def execute_directly(self, state):
        md_table_aligned = state["md_table_aligned"]
        llm = state["llm"]
        md_table_list = state["md_table_list"]
        caption = state["caption"]

        value_list = []
        round = 0
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        for md in md_table_list:
            round += 1
            self._step_output(f"Trial {round}")
            system_prompt = get_parameter_value_prompt(md_table_aligned, md, caption)
            agent = PKSumCommonAgent(llm=llm)
            res, processed_res, token_usage, reasoning_process = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=INSTRUCTION_PROMPT,
                schema=ParameterValueResult,
                post_process=post_process_matched_list,
                expected_rows=markdown_to_dataframe(md).shape[0],
            )
            self._step_output(
                state,
                step_reasoning_process=reasoning_process if reasoning_process is not None else "",
            )
            value_list.append(processed_res)
            total_token_usage = increase_token_usage(token_usage)

        return (
            ParameterValueResult(extracted_param_values=[[]]),
            value_list,
            total_token_usage,
            None,
        )

    def leave_step(self, state, step_reasoning_process, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["value_list"] = processed_res
            self._step_output(state, step_output="Result (value_list):")
            self._step_output(state, step_output=str(processed_res))
        return super().leave_step(state, step_reasoning_process, processed_res, token_usage)
