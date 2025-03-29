import pytest

from extractor.agents.pk_summary.pk_sum_drug_matching_step import (
    DrugMatchingAgentStep,
    DrugMatchingAutomaticStep
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

@pytest.mark.skip()
def test_DrugMatchingAgentStep(
    llm,
    md_table_drug,
    md_table_list,
    caption,
    md_table_aligned,
    step_callback,
):
    step = DrugMatchingAgentStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['md_table_drug'] = md_table_drug
    state['md_table_list'] = md_table_list
    state['md_table_aligned'] = md_table_aligned
    state['caption'] = caption
    state['step_callback'] = step_callback

    step.execute(state)

    assert state['drug_list'] is not None
    assert type(state['drug_list']) == list

@pytest.mark.skip()
def test_DrugMatchingAutomaticStep(
    llm,
    md_table_drug,
    md_table_list,
    caption,
    md_table_aligned,
    step_callback,
):
    step = DrugMatchingAutomaticStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['md_table_drug'] = md_table_drug
    state['md_table_list'] = md_table_list
    state['md_table_aligned'] = md_table_aligned
    state['caption'] = caption
    state['step_callback'] = step_callback

    step.execute(state)

    assert state['drug_list'] is not None
    assert type(state['drug_list']) == list

@pytest.mark.skip()
def test_DrugMatchingAutomaticStep_16143486_table_4(
    llm,
    md_table_drug_16143486_table_4,
    md_table_list_16143486_table_4,
    caption_16143486_table_4,
    md_table_aligned_16143486_table_4,
    step_callback,
):
    step = DrugMatchingAgentStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['md_table_drug'] = md_table_drug_16143486_table_4
    state['md_table_list'] = md_table_list_16143486_table_4
    state['md_table_aligned'] = md_table_aligned_16143486_table_4
    state['caption'] = caption_16143486_table_4
    state['step_callback'] = step_callback

    step.execute(state)

    assert state['drug_list'] is not None
    assert type(state['drug_list']) == list

def test_DrugMatchingAgentStep_34183327_table_2(
    llm,
    md_table_drug_34183327_table_2,
    md_table_list_34183327_table_2,
    caption_34183327_table_2,
    md_table_aligned_34183327_table_2,
    step_callback,
):
    step = DrugMatchingAgentStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['md_table_drug'] = md_table_drug_34183327_table_2
    state['md_table_list'] = md_table_list_34183327_table_2
    state['md_table_aligned'] = md_table_aligned_34183327_table_2
    state['caption'] = caption_34183327_table_2
    state['step_callback'] = step_callback

    step.execute(state)

    assert state['drug_list'] is not None
    assert type(state['drug_list']) == list

def test_DrugMatchingAgentStep_22050870_table_3(
    llm,
    md_table_drug_34183327_table_2,
    md_table_list_34183327_table_2,
    caption_34183327_table_2,
    md_table_aligned_34183327_table_2,
    step_callback,
):
    step = DrugMatchingAgentStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['md_table_drug'] = md_table_drug_34183327_table_2
    state['md_table_list'] = md_table_list_34183327_table_2
    state['md_table_aligned'] = md_table_aligned_34183327_table_2
    state['caption'] = caption_34183327_table_2
    state['step_callback'] = step_callback

    step.execute(state)

    assert state['drug_list'] is not None
    assert type(state['drug_list']) == list

