import pytest

from extractor.agents.pk_individual.pk_ind_drug_matching_step import DrugMatchingAgentStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

@pytest.mark.skip()
def test_DrugMatchingAgentStep_29100749_table_2(
    llm, 
    md_table_aligned_29100749_table_2, 
    md_table_drug_29100749_table_2, 
    caption_29100749_table_2, 
    step_callback,
    md_table_list_29100749_table_2,
):
    step = DrugMatchingAgentStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["md_table_aligned"] = md_table_aligned_29100749_table_2
    state["md_table_drug"] = md_table_drug_29100749_table_2
    state["caption"] = caption_29100749_table_2
    state["md_table_list"] = md_table_list_29100749_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["drug_list"] is not None
    assert type(state["drug_list"]) == list

def test_DrugMatchingAgentStep_18426260_table_0(
    llm,
    md_table_aligned_18426260_table_0,
    md_table_drug_18426260_table_0,
    caption_18426260_table_0,
    step_callback,
    md_table_list_18426260_table_0,
):
    step = DrugMatchingAgentStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["md_table_aligned"] = md_table_aligned_18426260_table_0
    state["md_table_drug"] = md_table_drug_18426260_table_0
    state["caption"] = caption_18426260_table_0
    state["md_table_list"] = md_table_list_18426260_table_0
    state["step_callback"] = step_callback
    
    step.execute(state)

    assert state["drug_list"] is not None
    assert type(state["drug_list"]) == list

