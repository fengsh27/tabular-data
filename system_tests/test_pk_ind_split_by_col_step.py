import pytest

from extractor.agents.pk_individual.pk_ind_split_by_col_step import SplitByColumnsStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

@pytest.mark.skip
def test_SplitByColumnsStep_29100749_table_2(
    llm, 
    step_callback,
    md_table_aligned_29100749_table_2, 
    col_mapping_29100749_table_2
):
    step = SplitByColumnsStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["col_mapping"] = col_mapping_29100749_table_2
    state["md_table_aligned"] = md_table_aligned_29100749_table_2
    state["step_callback"] = step_callback
    
    step.execute(state)

    assert state["md_table_list"] is not None
    assert type(state["md_table_list"]) == list

def test_SplitByColumnsStep_23200982_table_2(
    llm, 
    step_callback,
    md_table_aligned_23200982_table_2, 
    col_mapping_23200982_table_2,
):
    step = SplitByColumnsStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["col_mapping"] = col_mapping_23200982_table_2
    state["md_table_aligned"] = md_table_aligned_23200982_table_2
    state["step_callback"] = step_callback
    
    step.execute(state)

    assert state["md_table_list"] is not None
    assert type(state["md_table_list"]) == list

