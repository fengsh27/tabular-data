from extractor.agents.pe_study_outcome.pe_study_out_common_step import PEStudyOutCommonAgentStep
from extractor.agents.pe_study_outcome.pe_study_out_common_agent import PEStudyOutCommonAgent
from extractor.agents.pe_study_outcome.pe_study_out_row_categorize_agent import (
    RowCategorizeResult,
    get_row_categorize_prompt,
    RowCategorizeJsonSchema,
    post_process_validate_categorized_result,
)
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage

import logging
logger = logging.getLogger(__name__)


class RowCategorizeStep(PEStudyOutCommonAgentStep):
    def __init__(self):
        super().__init__()

        self.start_title = "Categorizing Rows"
        self.end_title = "Completed to Categorize Rows"

    def get_system_prompt(self, state):
        md_table = state["md_table"]
        system_prompt = get_row_categorize_prompt(md_table)
        previous_errors_prompt = self._get_previous_errors_prompt(state)
        return system_prompt + previous_errors_prompt

    def get_schema(self):
        return RowCategorizeJsonSchema

    def get_post_processor_and_kwargs(self, state):
        md_table = state["md_table"]
        return post_process_validate_categorized_result, {
            "md_table": md_table
        }

    def execute_directly(self, state):
        llm = state["llm"]
        col_mapping = state["col_mapping"]
        row_headers_keys = [k for k, v in col_mapping.items() if v == "Row headers"]
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        return_dict = {}
        for key in row_headers_keys:
            agent = PEStudyOutCommonAgent(llm=llm)
            md_table = state["md_table"]
            schema = self.get_schema()
            system_prompt = get_row_categorize_prompt(
                md_table,
                key
            )
            instruction_prompt = self.get_instruction_prompt(state)

            res, processed_res, token_usage = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
                post_process=post_process_validate_categorized_result,
                md_table=md_table,
            )

            self._step_output(
                state,
                step_reasoning_process=res['reasoning_process']
                if res is not None
                else "",
            )
            total_token_usage = increase_token_usage(
                total_token_usage, token_usage
            )
            return_dict[key] = res['categorized_headers']

        return (
            None,
            return_dict,
            total_token_usage,
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["row_mapping"] = processed_res
            self._step_output(state, step_output="Result (row_mapping):")
            self._step_output(state, step_output=str(processed_res))
        return super().leave_step(state, res, processed_res, token_usage)
