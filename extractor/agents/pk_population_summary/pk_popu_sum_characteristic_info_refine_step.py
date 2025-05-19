from extractor.agents.pk_population_summary.pk_popu_sum_common_step import PKPopuSumCommonAgentStep
from extractor.agents.pk_population_summary.pk_popu_sum_common_agent import (
    PKPopuSumCommonAgentResult,
)
from extractor.agents.pk_population_summary.pk_popu_sum_workflow_utils import PKPopuSumWorkflowState

from extractor.agents.pk_population_summary.pk_popu_sum_characteristic_info_refine_agent import (
    CharacteristicInfoRefinedResult,
    get_characteristic_info_refine_prompt,
    post_process_refined_characteristic_info,
)


class CharacteristicInfoRefinementStep(PKPopuSumCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Refining Drug Information"
        self.end_title = "Completed to Refine Drug Information"

    def get_system_prompt(self, state: PKPopuSumWorkflowState):
        title = state["title"]
        full_text = state["full_text"]
        md_table_characteristic = state["md_table_characteristic"]
        return get_characteristic_info_refine_prompt(title, full_text, md_table_characteristic)

    def leave_step(
        self,
        state: PKPopuSumWorkflowState,
        res: PKPopuSumCommonAgentResult,
        processed_res=None,
        token_usage=None,
    ):
        if processed_res is not None:
            state["md_table_characteristic_refined"] = processed_res
            self._step_output(state, step_output="Result (md_table_characteristic_refined):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, res, processed_res, token_usage)

    def get_schema(self):
        return CharacteristicInfoRefinedResult

    def get_post_processor_and_kwargs(self, state: PKPopuSumWorkflowState):
        md_table_characteristic = state["md_table_characteristic"]
        return post_process_refined_characteristic_info, {"md_table_characteristic": md_table_characteristic}
