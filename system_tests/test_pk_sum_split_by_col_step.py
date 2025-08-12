import pytest

from extractor.agents.pk_summary.pk_sum_split_by_col_step import SplitByColumnsStep
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState


@pytest.mark.skip()
def test_SplitByColumnsStep(llm, md_table_aligned, col_mapping):
    # res = SplitByColumnsResult(re)
    step = SplitByColumnsStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["col_mapping"] = col_mapping
    state["md_table_aligned"] = md_table_aligned

    step.execute(state)

    assert state["md_table_list"] is not None
    assert type(state["md_table_list"]) == list


# @pytest.mark.skip()
def test_SplitByColumnsStep_29943508(
    llm, md_table_aligned_29943508, col_mapping_29943508
):
    # res = SplitByColumnsResult(re)
    step = SplitByColumnsStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["col_mapping"] = col_mapping_29943508
    state["md_table_aligned"] = md_table_aligned_29943508

    step.execute(state)

    assert state["md_table_list"] is not None
    assert type(state["md_table_list"]) == list


@pytest.mark.skip()
def test_SplitByColumnsStep_28794838_table_2(
    llm, md_table_aligned_28794838_table_2, col_mapping_28794838_table_2, step_callback
):
    # res = SplitByColumnsResult(re)
    step = SplitByColumnsStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["col_mapping"] = col_mapping_28794838_table_2
    state["md_table_aligned"] = md_table_aligned_28794838_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["md_table_list"] is not None
    assert type(state["md_table_list"]) == list
