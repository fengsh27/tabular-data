import pytest

from extractor.agents.pk_summary.pk_sum_header_categorize_step import (
    HeaderCategorizeStep
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

# @pytest.mark.skip()
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
    state['llm'] = llm
    state['md_table_drug'] = md_table_drug
    state['md_table_list'] = md_table_list
    state['md_table_aligned'] = md_table_aligned
    state['caption'] = caption
    state['step_callback'] = step_callback

    step.execute(state)

    assert state['col_mapping'] is not None
    assert type(state['col_mapping']) == list

