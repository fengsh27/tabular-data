import pytest

from extractor.agents.pk_summary.pk_sum_split_by_col_agent import SplitByColumnsResult
from extractor.agents.pk_summary.pk_sum_split_by_col_step import (
    SplitByColumnsStep
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

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