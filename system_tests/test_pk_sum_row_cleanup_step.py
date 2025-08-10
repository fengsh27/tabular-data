import pytest
import pandas as pd
from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pk_summary.pk_sum_row_cleanup_step import RowCleanupStep
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState
from system_tests.conftest import caption_17635501_table_3

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

def test_RowCleanupStep_on_17635501(
    llm,
    df_combined_17635501_table_3,
    caption_17635501_table_3,
    step_callback,
):
    step = RowCleanupStep()
    state = PKSumWorkflowState(
        llm=llm,
        df_combined=df_combined_17635501_table_3,
        caption=caption_17635501_table_3,
        step_callback=step_callback,
    )

    state = step.execute(state)

    assert state["df_combined"] is not None
    assert isinstance(state["df_combined"], pd.DataFrame)

    md_table_combined = dataframe_to_markdown(state["df_combined"])
