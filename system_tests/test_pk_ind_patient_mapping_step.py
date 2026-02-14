import pytest

from extractor.agents.pk_individual.pk_ind_patient_matching_step import PatientMatchingAgentStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

@pytest.mark.skip()
def test_PatientMatchingAgentStep_29100749_table_2(
    llm, 
    md_table_aligned_29100749_table_2, 
    md_table_patient_29100749_table_2,
    md_table_patient_refined_29100749_table_2, 
    caption_29100749_table_2, 
    step_callback,
    md_table_list_29100749_table_2,
):
    step = PatientMatchingAgentStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["md_table_aligned"] = md_table_aligned_29100749_table_2
    state["md_table_patient"] = md_table_patient_29100749_table_2
    state["md_table_patient_refined"] = md_table_patient_refined_29100749_table_2
    state["caption"] = caption_29100749_table_2
    state["md_table_list"] = md_table_list_29100749_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["patient_list"] is not None
    assert type(state["patient_list"]) == list

@pytest.mark.skip()
def test_PatientMatchingAgentStep_33253437_table_0(
    llm,
    llm_agent,
    md_table_aligned_table_0_33253437,
    md_table_patient_table_0_33253437,
    md_table_patient_refined_table_0_33253437,
    caption_table_0_33253437,
    step_callback,
    md_table_list_table_0_33253437,
):
    step = PatientMatchingAgentStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["llm2"] = llm_agent
    state["md_table_aligned"] = md_table_aligned_table_0_33253437
    state["md_table_patient"] = md_table_patient_table_0_33253437
    state["md_table_patient_refined"] = md_table_patient_refined_table_0_33253437
    state["caption"] = caption_table_0_33253437
    state["md_table_list"] = md_table_list_table_0_33253437
    state["step_callback"] = step_callback

    step.execute(state)
    assert state["patient_list"] is not None
    assert type(state["patient_list"]) == list

def test_PatientMatchingAgentStep_18426260_table_0(
    llm,
    llm_agent,
    md_table_aligned_18426260_table_0,
    pk_ind_md_table_patient_18426260_table_0,
    md_table_patient_refined_18426260_table_0,
    caption_18426260_table_0,
    md_table_list_18426260_table_0,
    step_callback,
):
    step = PatientMatchingAgentStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["llm2"] = llm_agent
    state["md_table_aligned"] = md_table_aligned_18426260_table_0
    state["md_table_patient"] = pk_ind_md_table_patient_18426260_table_0
    state["md_table_patient_refined"] = md_table_patient_refined_18426260_table_0
    state["caption"] = caption_18426260_table_0
    state["md_table_list"] = md_table_list_18426260_table_0
    state["step_callback"] = step_callback

    step.execute(state)
    
    assert state["patient_list"] is not None
    assert type(state["patient_list"]) == list


