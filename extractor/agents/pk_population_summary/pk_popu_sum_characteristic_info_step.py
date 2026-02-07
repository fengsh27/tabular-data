from extractor.agents.pk_population_summary.pk_popu_sum_common_step import PKPopuSumCommonAgentStep
from extractor.agents.pk_population_summary.pk_popu_sum_common_agent import (
    PKPopuSumCommonAgentResult,
)
from extractor.agents.pk_population_summary.pk_popu_sum_characteristic_info_agent import (
    CHARACTERISTIC_INFO_PROMPT,
    CharacteristicInfoResult,
    post_process_characteristic_info,
)
from extractor.agents.pk_population_summary.pk_popu_sum_workflow_utils import PKPopuSumWorkflowState


class CharacteristicInfoExtractionStep(PKPopuSumCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Extracting Characteristic Information"
        self.end_title = "Completed to Extract Characteristic Information"

    def get_system_prompt(self, state):
        title = state["title"]
        full_text = state["full_text"]
        system_prompt = CHARACTERISTIC_INFO_PROMPT.format(
            title=title,
            full_text=full_text,
        )
        previous_errors_prompt = self._get_previous_errors_prompt(state)
        return system_prompt + previous_errors_prompt

    def leave_step(
        self,
        state: PKPopuSumWorkflowState,
        res: PKPopuSumCommonAgentResult,
        processed_res=None,
        token_usage=None,
    ):
        if processed_res is not None:
            state["md_table_characteristic"] = processed_res
            self._step_output(state, step_output="Result (md_table_characteristic):")
            self._step_output(state, step_output=processed_res)
        super().leave_step(state, res, processed_res, token_usage)

    def get_schema(self):
        return CharacteristicInfoResult

    def get_post_processor_and_kwargs(self, state):
        return post_process_characteristic_info, None
