import pytest
from extractor.agents.pk_population_summary.pk_popu_sum_characteristic_info_step import (
    CharacteristicInfoExtractionStep,
    PKPopuSumWorkflowState,
)

from system_tests.conftest_data_18391836 import title, full_text

def test_characteristic_info_step(llm, step_callback):
    step = CharacteristicInfoExtractionStep()
    state = PKPopuSumWorkflowState()
    state["llm"] = llm
    state["step_callback"] = step_callback
    state["title"] = title
    state["full_text"] = full_text

    state = step.execute(state)
    assert state is not None
