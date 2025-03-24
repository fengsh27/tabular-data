
from extractor.agents.pk_summary.pk_sum_drug_matching_step import (
    DrugMatchingStep
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

def test_DrugMatchingStep(
    llm,
    md_table_drug,
    md_table_list,
    caption,
    md_table_aligned,
):
    step = DrugMatchingStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['md_table_drug'] = md_table_drug
    state['md_table_list'] = md_table_list
    state['md_table_aligned'] = md_table_aligned
    state['caption'] = caption

    step.execute(state)

    assert state['drug_list'] is not None
    assert type(state['drug_list']) == list

