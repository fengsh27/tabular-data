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

