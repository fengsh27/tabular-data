from extractor.agents.pk_individual.pk_ind_common_step import PKIndCommonAgentStep
from extractor.agents.pk_individual.pk_ind_param_type_align_agent import (
    ParameterTypeAlignResult,
    get_parameter_type_align_prompt,
    post_process_parameter_type_align,
)


class ParametertypeAlignStep(PKIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Aligning Parameter Type"
        self.end_title = "Completed to Align Parameter Type"

    def get_system_prompt(self, state):
        md_table_individual = state["md_table_individual"]
        return get_parameter_type_align_prompt(md_table_individual)

    def get_schema(self):
        return ParameterTypeAlignResult

    def get_post_processor_and_kwargs(self, state):
        md_table_individual = state["md_table_individual"]
        return post_process_parameter_type_align, {"md_table_individual": md_table_individual}

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["md_table_aligned"] = processed_res
            self._step_output(state, step_output="Result (md_table_aligned):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, res, processed_res, token_usage)
