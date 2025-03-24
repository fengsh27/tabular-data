import pandas as pd

from extractor.agents.pk_summary.pk_sum_time_unit_step import (
    TimeExtractionStep
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import (
    PKSumWorkflowState
)

def test_TimeExtractionStep(
    llm,
    md_table_aligned,
    df_combined,
    caption,
):
    step = TimeExtractionStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['df_combined'] = df_combined
    state['md_table_aligned'] = md_table_aligned
    state['caption'] = caption

    step.execute(state)

    assert state['df_combined'] is not None
    assert isinstance(state['df_combined'], pd.DataFrame)



