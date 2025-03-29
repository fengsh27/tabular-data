import pytest

from extractor.agents.pk_summary.pk_sum_patient_matching_step import (
    PatientMatchingAutomaticStep,
    PatientMatchingAgentStep,
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

@pytest.mark.skip()
def test_PatientMatchingAutomaticStep_30825333_table_2(
        llm, 
        md_table_patient_refined_30825333_table_2, 
        md_table_list_30825333_table_2,
        caption_30825333_table_2,
        step_callback
    ):
    step = PatientMatchingAutomaticStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['md_table_patient_refined'] = md_table_patient_refined_30825333_table_2
    state['md_table_list'] = md_table_list_30825333_table_2
    state['caption'] = caption_30825333_table_2
    state['step_callback'] = step_callback

    step.execute(state)

    assert state['patient_list'] is not None
    assert type(state['patient_list']) == list


def test_PatientMatchingAgentStep_35489632_table_2(
        llm, 
        md_table_patient_refined_35489632_table_2, 
        md_table_list_35489632_table_2,
        md_table_aligned_35489632_table_2,
        md_table_patient_35489632_table_2,
        caption_35489632_table_2,
        step_callback
    ):
    step = PatientMatchingAgentStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['md_table_aligned'] = md_table_aligned_35489632_table_2
    state['md_table_patient'] = md_table_patient_35489632_table_2
    state['md_table_patient_refined'] = md_table_patient_refined_35489632_table_2
    state['md_table_list'] = md_table_list_35489632_table_2
    state['caption'] = caption_35489632_table_2
    state['step_callback'] = step_callback

    step.execute(state)

    assert state['patient_list'] is not None
    assert type(state['patient_list']) == list