import pytest

from extractor.agents.pk_summary.pk_sum_split_by_col_agent import SplitByColumnsResult
from extractor.agents.pk_summary.pk_sum_split_by_col_step import (
    SplitByColumnsStep
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

@pytest.mark.skip()
def test_SplitByColumnsStep(llm, md_table_aligned, col_mapping):
    # res = SplitByColumnsResult(re)
    step = SplitByColumnsStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state["col_mapping"] = col_mapping
    state["md_table_aligned"] = md_table_aligned

    step.execute(state)

    assert state["md_table_list"] is not None
    assert type(state["md_table_list"]) == list

def test_SplitByColumnsStep_29943508(llm, md_table_aligned_29943508, col_mapping_29943508):
    # res = SplitByColumnsResult(re)
    step = SplitByColumnsStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state["col_mapping"] = col_mapping_29943508
    state["md_table_aligned"] = md_table_aligned_29943508

    step.execute(state)

    assert state["md_table_list"] is not None
    assert type(state["md_table_list"]) == list