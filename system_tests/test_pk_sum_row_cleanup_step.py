import pytest
import pandas as pd
from extractor.agents.pk_summary.pk_sum_row_cleanup_step import RowCleanupStep
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

@pytest.mark.skip(reason="skip due to token usage")
def test_RowCleanupStep(
    llm,
    md_table_aligned,
    df_combined,
    caption,
):
    step = RowCleanupStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["df_combined"] = df_combined
    state["md_table_aligned"] = md_table_aligned
    state["caption"] = caption

    step.execute(state)

    assert state["df_combined"] is not None
    assert isinstance(state["df_combined"], pd.DataFrame)
