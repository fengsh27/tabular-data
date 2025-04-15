from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonAgentStep
from extractor.agents.pk_summary.pk_sum_param_type_align_agent import (
    ParameterTypeAlignResult,
    get_parameter_type_align_prompt,
    post_process_parameter_type_align,
)


class ParametertypeAlignStep(PKSumCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Aligning Parameter Type"
        self.end_title = "Completed to Align Parameter Type"

    def get_system_prompt(self, state):
        md_table_summary = state["md_table_summary"]
        return get_parameter_type_align_prompt(md_table_summary)

    def get_schema(self):
        return ParameterTypeAlignResult

    def get_post_processor_and_kwargs(self, state):
        md_table_summary = state["md_table_summary"]
        return post_process_parameter_type_align, {"md_table_summary": md_table_summary}

    def leave_step(self, state, step_reasoning_process, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["md_table_aligned"] = processed_res
            self._step_output(state, step_output="Result (md_table_aligned):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, step_reasoning_process, processed_res, token_usage)
