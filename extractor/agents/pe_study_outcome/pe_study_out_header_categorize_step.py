from extractor.agents.pe_study_outcome.pe_study_out_common_step import PEStudyOutCommonAgentStep
from extractor.agents.pe_study_outcome.pe_study_out_header_categorize_agent import (
    HeaderCategorizeResult,
    get_header_categorize_prompt,
    HeaderCategorizeJsonSchema,
    post_process_validate_categorized_result,
)


class HeaderCategorizeStep(PEStudyOutCommonAgentStep):
    def __init__(self):
        super().__init__()

        self.start_title = "Categorizing Column Header"
        self.end_title = "Completed to Categorize Column Header"

    def get_system_prompt(self, state):
        md_table = state["md_table"]
        system_prompt = get_header_categorize_prompt(md_table)
        previous_errors_prompt = self._get_previous_errors_prompt(state)
        return system_prompt + previous_errors_prompt

    def get_schema(self):
        return HeaderCategorizeJsonSchema

    def get_post_processor_and_kwargs(self, state):
        md_table = state["md_table"]
        return post_process_validate_categorized_result, {
            "md_table": md_table
        }

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        result: HeaderCategorizeResult = processed_res
        if result is not None and result.categorized_headers is not None:
            state["col_mapping"] = result.categorized_headers
            self._step_output(state, step_output="Result (col_mapping):")
            self._step_output(state, step_output=str(result.categorized_headers))
        return super().leave_step(state, res, processed_res, token_usage)
