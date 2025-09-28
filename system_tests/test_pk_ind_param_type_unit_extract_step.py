import pytest

from extractor.agents.pk_individual.pk_ind_param_type_unit_extract_step import ExtractParamTypeAndUnitStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

def test_ExtractParamTypeAndUnitStep_29100749_table_2(
    llm, 
    md_table_aligned_29100749_table_2, 
    col_mapping_29100749_table_2, 
    caption_29100749_table_2,
    md_table_list_29100749_table_2,
    step_callback
):
    step = ExtractParamTypeAndUnitStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["col_mapping"] = col_mapping_29100749_table_2
    state["md_table_aligned"] = md_table_aligned_29100749_table_2
    state["caption"] = caption_29100749_table_2
    state["md_table_list"] = md_table_list_29100749_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["type_unit_list"] is not None
    assert type(state["type_unit_list"]) == list