import pytest
import logging
from extractor.agents.pk_summary.pk_sum_param_value_step import (
    ParameterValueExtractionStep,
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

logger = logging.getLogger(__name__)


@pytest.mark.skip()
def test_ParamterValueExtractionStep(
    llm, caption, md_table_aligned, md_table_list, step_callback
):
    step = ParameterValueExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption
    state["md_table_aligned"] = md_table_aligned
    state["md_table_list"] = md_table_list
    state["step_callback"] = step_callback

    updated_state = step.execute(state)

    assert updated_state["value_list"] is not None
    assert type(updated_state["value_list"]) == list


@pytest.mark.skip()
def test_ParameterValueExtractionStep1(
    llm, caption1, md_table_aligned1, md_table_list1, step_callback
):
    step = ParameterValueExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption1
    state["md_table_aligned"] = md_table_aligned1
    state["md_table_list"] = md_table_list1
    state["step_callback"] = step_callback

    updated_state = step.execute(state)

    assert updated_state["value_list"] is not None
    assert type(updated_state["value_list"]) == list

@pytest.mark.skip()
def test_ParameterValueExtractionStep_22050870_table_2(
    llm,
    caption_22050870_table_2,
    md_table_aligned_22050870_table_2,
    md_table_list_22050870_table_2,
    step_callback,
):
    step = ParameterValueExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption_22050870_table_2
    state["md_table_aligned"] = md_table_aligned_22050870_table_2
    state["md_table_list"] = md_table_list_22050870_table_2
    state["step_callback"] = step_callback

    updated_state = step.execute(state)

    assert updated_state["value_list"] is not None
    assert type(updated_state["value_list"]) == list

@pytest.mark.skip()
def test_ParameterValueExtractionStep_35465728_table_2(
    llm,
    caption_35465728_table_2,
    md_table_aligned_35465728_table_2,
    md_table_list_35465728_table_2,
    step_callback,
):
    step = ParameterValueExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption_35465728_table_2
    state["md_table_aligned"] = md_table_aligned_35465728_table_2
    state["md_table_list"] = md_table_list_35465728_table_2
    state["step_callback"] = step_callback

    updated_state = step.execute(state)

    assert updated_state["value_list"] is not None
    assert type(updated_state["value_list"]) == list


def test_ParameterValueExtractionStep_17635501_table_3(
    llm,
    caption_17635501_table_3,
    md_table_aligned_17635501_table_3,
    md_table_list_17635501_table_3,
    step_callback,
):
    step = ParameterValueExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption_17635501_table_3
    state["md_table_aligned"] = md_table_aligned_17635501_table_3
    state["md_table_list"] = md_table_list_17635501_table_3
    state["step_callback"] = step_callback

    updated_state = step.execute(state)

    assert updated_state["value_list"] is not None
    assert type(updated_state["value_list"]) == list