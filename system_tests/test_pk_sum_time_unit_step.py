import pandas as pd
import pytest

from extractor.agents.pk_summary.pk_sum_time_unit_step import TimeExtractionStep
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState


@pytest.mark.skip()
def test_TimeExtractionStep(
    llm,
    md_table_aligned,
    df_combined,
    caption,
):
    step = TimeExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["df_combined"] = df_combined
    state["md_table_aligned"] = md_table_aligned
    state["caption"] = caption

    step.execute(state)

    assert state["df_combined"] is not None
    assert isinstance(state["df_combined"], pd.DataFrame)


def test_TimeExtractionStep_30825333_table_2(
    llm,
    md_table_aligned_30825333_table_2,
    df_combined_30825333_table_2,
    caption_30825333_table_2,
):
    step = TimeExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["df_combined"] = df_combined_30825333_table_2
    state["md_table_aligned"] = md_table_aligned_30825333_table_2
    state["caption"] = caption_30825333_table_2

    step.execute(state)

    assert state["df_combined"] is not None
    assert isinstance(state["df_combined"], pd.DataFrame)
