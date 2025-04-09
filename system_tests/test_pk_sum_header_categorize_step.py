import pytest

from extractor.agents.pk_summary.pk_sum_header_categorize_step import (
    HeaderCategorizeStep,
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState


@pytest.mark.skip()
def test_HeaderCategorizeStep(
    llm,
    md_table_drug,
    md_table_list,
    caption,
    md_table_aligned,
    step_callback,
):
    step = HeaderCategorizeStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_drug"] = md_table_drug
    state["md_table_list"] = md_table_list
    state["md_table_aligned"] = md_table_aligned
    state["caption"] = caption
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["col_mapping"] is not None
    assert type(state["col_mapping"]) == dict


@pytest.mark.skip()
def test_HeaderCategorizeStep_30825333_table_2(
    llm,
    md_table_drug_30825333_table_2,
    caption_30825333_table_2,
    md_table_aligned_30825333_table_2,
    step_callback,
):
    step = HeaderCategorizeStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_drug"] = md_table_drug_30825333_table_2
    state["md_table_aligned"] = md_table_aligned_30825333_table_2
    state["caption"] = caption_30825333_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["col_mapping"] is not None
    assert type(state["col_mapping"]) == dict

@pytest.mark.skip()
def test_HeaderCategorizeStep_28794837_table_2(
    llm,
    md_table_drug_28794838_table_2,
    caption_28794838_table_2,
    md_table_aligned_28794838_table_2,
    step_callback,
):
    step = HeaderCategorizeStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_drug"] = md_table_drug_28794838_table_2
    state["md_table_aligned"] = md_table_aligned_28794838_table_2
    state["caption"] = caption_28794838_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["col_mapping"] is not None
    assert type(state["col_mapping"]) == dict

def test_HeaderCategorizeStep_29943508_table_1(
    llm,
    md_table_aligned_29943508_table_1,
    caption_29943508,
    md_table_drug_29943508_table_1,
):
    step = HeaderCategorizeStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption_29943508
    state["md_table_drug"] = md_table_drug_29943508_table_1
    state["md_table_aligned"] = md_table_aligned_29943508_table_1

    state = step.execute(state)

    assert state["col_mapping"] is not None
    assert type(state["col_mapping"]) == dict