import pytest

from extractor.agents.pk_summary.pk_sum_param_type_unit_extract_step import (
    ExtractParamTypeAndUnitStep,
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState


@pytest.mark.skip()
def test_ExtractParamTypeAndUnitStep(
    llm, col_mapping, md_table_aligned, caption, md_table_list
):
    step = ExtractParamTypeAndUnitStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["col_mapping"] = col_mapping
    state["md_table_aligned"] = md_table_aligned
    state["md_table_list"] = md_table_list
    state["caption"] = caption

    step.execute(state)

    assert state["type_unit_list"] is not None
    assert type(state["type_unit_list"]) == list

@pytest.mark.skip()
def test_ExtractParamTypeAndUnitStep_30825333_table_2(
    llm,
    col_mapping_30825333_table_2,
    md_table_aligned_30825333_table_2,
    caption_30825333_table_2,
    md_table_list_30825333_table_2,
):
    step = ExtractParamTypeAndUnitStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["col_mapping"] = col_mapping_30825333_table_2
    state["md_table_aligned"] = md_table_aligned_30825333_table_2
    state["md_table_list"] = md_table_list_30825333_table_2
    state["caption"] = caption_30825333_table_2

    step.execute(state)

    assert state["type_unit_list"] is not None
    assert type(state["type_unit_list"]) == list

def test_ExtractParamTypeAndUnitStep_35465728_table_2(
    llm,
    col_mapping_35465728_table_2,
    md_table_aligned_35465728_table_2,
    caption_35465728_table_2,
    md_table_list_35465728_table_2,
):
    step = ExtractParamTypeAndUnitStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["col_mapping"] = col_mapping_35465728_table_2
    state["md_table_aligned"] = md_table_aligned_35465728_table_2
    state["md_table_list"] = md_table_list_35465728_table_2
    state["caption"] = caption_35465728_table_2

    step.execute(state)

    assert state["type_unit_list"] is not None
    assert type(state["type_unit_list"]) == list
