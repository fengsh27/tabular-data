import pytest

from extractor.agents.pk_summary.pk_sum_individual_data_del_step import (
    IndividualDataDelStep
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

# @pytest.mark.skip()
def test_HeaderCategorizeStep(
    llm,
    md_table_16143486_table_4,
    md_table_drug_16143486_table_4,
    caption_16143486_table_4,
    md_table_patient_16143486_table_4,
    md_table_patient_refined_16143486_table_4,
    step_callback,
):
    step = IndividualDataDelStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['md_table'] = md_table_16143486_table_4
    state['caption'] = caption_16143486_table_4
    state['step_callback'] = step_callback

    step.execute(state)

    assert state['md_table_summary'] is not None
    assert type(state['md_table_summary']) == str

