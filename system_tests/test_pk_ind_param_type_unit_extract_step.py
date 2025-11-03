import pytest

from extractor.agents.pk_individual.pk_ind_param_type_unit_extract_step import ExtractParamTypeAndUnitStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

@pytest.mark.skip()
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



def test_ExtractParamTypeAndUnitStep_33253437_table_0(
    llm,
    llm_agent,
    md_table_aligned_table_0_33253437,
    col_mapping_table_0_33253437,
    caption_table_0_33253437,
    md_table_list_table_0_33253437,
    step_callback
):
    step = ExtractParamTypeAndUnitStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["llm2"] = llm_agent
    state["col_mapping"] = col_mapping_table_0_33253437
    state["md_table_aligned"] = md_table_aligned_table_0_33253437
    state["caption"] = caption_table_0_33253437
    state["md_table_list"] = md_table_list_table_0_33253437
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["type_unit_list"] is not None
    assert type(state["type_unit_list"]) == list