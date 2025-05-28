from extractor.agents.pk_population_individual.pk_popu_ind_common_step import PKPopuIndCommonAgentStep
from extractor.agents.pk_population_individual.pk_popu_ind_common_agent import (
    PKPopuIndCommonAgentResult,
)
from extractor.agents.pk_population_individual.pk_popu_ind_workflow_utils import PKPopuIndWorkflowState

from extractor.agents.pk_population_individual.pk_popu_ind_characteristic_info_refine_agent import (
    CharacteristicInfoRefinedResult,
    get_characteristic_info_refine_prompt,
    post_process_refined_characteristic_info,
)


class CharacteristicInfoRefinementStep(PKPopuIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Refining Drug Information"
        self.end_title = "Completed to Refine Drug Information"

    def get_system_prompt(self, state: PKPopuIndWorkflowState):
        title = state["title"]
        full_text = state["full_text"]
        md_table_characteristic = state["md_table_characteristic"]
        return get_characteristic_info_refine_prompt(title, full_text, md_table_characteristic)

    def leave_step(
        self,
        state: PKPopuIndWorkflowState,
        res: PKPopuIndCommonAgentResult,
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

    def get_post_processor_and_kwargs(self, state: PKPopuIndWorkflowState):
        md_table_characteristic = state["md_table_characteristic"]
        return post_process_refined_characteristic_info, {"md_table_characteristic": md_table_characteristic}
