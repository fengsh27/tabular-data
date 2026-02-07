import pytest

from extractor.agents.pk_summary.pk_sum_patient_matching_step import (
    PatientMatchingAutomaticStep,
    PatientMatchingAgentStep,
)
from extractor.agents.pk_summary.pk_sum_patient_matching_agent import (
    MatchedPatientResult
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState


@pytest.mark.skip()
def test_PatientMatchingAutomaticStep_30825333_table_2(
    llm,
    md_table_patient_refined_30825333_table_2,
    md_table_list_30825333_table_2,
    caption_30825333_table_2,
    step_callback,
):
    step = PatientMatchingAutomaticStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_patient_refined"] = md_table_patient_refined_30825333_table_2
    state["md_table_list"] = md_table_list_30825333_table_2
    state["caption"] = caption_30825333_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["patient_list"] is not None
    assert type(state["patient_list"]) == list

@pytest.mark.skip()
def test_PatientMatchingAgentStep_35489632_table_2(
    llm,
    md_table_patient_refined_35489632_table_2,
    md_table_list_35489632_table_2,
    md_table_aligned_35489632_table_2,
    md_table_patient_35489632_table_2,
    caption_35489632_table_2,
    step_callback,
):
    step = PatientMatchingAgentStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_aligned"] = md_table_aligned_35489632_table_2
    state["md_table_patient"] = md_table_patient_35489632_table_2
    state["md_table_patient_refined"] = md_table_patient_refined_35489632_table_2
    state["md_table_list"] = md_table_list_35489632_table_2
    state["caption"] = caption_35489632_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["patient_list"] is not None
    assert type(state["patient_list"]) == list

@pytest.mark.skip()
def test_PatientMatchingAgentStep_22050870_table_3(
    llm,
    md_table_patient_refined_22050870_table_3,
    md_table_list_22050870_table_3,
    caption_22050870_table_3,
    md_table_aligned_22050870_table_3,
    md_table_patient_22050870_table_3,
    step_callback,
):
    step = PatientMatchingAgentStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_patient_refined"] = md_table_patient_refined_22050870_table_3
    state["md_table_list"] = md_table_list_22050870_table_3
    state["caption"] = caption_22050870_table_3
    state["step_callback"] = step_callback
    state["md_table_aligned"] = md_table_aligned_22050870_table_3
    state["md_table_patient"] = md_table_patient_22050870_table_3

    step.execute(state)

    assert state["patient_list"] is not None
    assert type(state["patient_list"]) == list

@pytest.mark.skip()
def test_PatientMatchingAgentStep_17635501_table_3(
    llm,
    md_table_patient_refined_17635501_table_3,
    md_table_list_17635501_table_3,
    caption_17635501_table_3,
    md_table_aligned_17635501_table_3,
    md_table_patient_17635501_table_3,
    step_callback,
):
    step = PatientMatchingAgentStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_patient_refined"] = md_table_patient_refined_17635501_table_3
    state["md_table_list"] = md_table_list_17635501_table_3
    state["caption"] = caption_17635501_table_3
    state["step_callback"] = step_callback
    state["md_table_aligned"] = md_table_aligned_17635501_table_3
    state["md_table_patient"] = md_table_patient_17635501_table_3

    step.execute(state)

    assert state["patient_list"] is not None
    assert type(state["patient_list"]) == list

@pytest.mark.skip()
def test_PatientMatchingAgentStep_34183327_table_2(
    llm,
    md_table_patient_refined_34183327_table_2,
    md_table_list_34183327_table_2,
    caption_34183327_table_2,
    md_table_aligned_34183327_table_2,
    md_table_patient_34183327_table_2,
    step_callback,
):
    json_obj = MatchedPatientResult.model_json_schema()
    print(json_obj)

    step = PatientMatchingAgentStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_patient_refined"] = md_table_patient_refined_34183327_table_2
    state["md_table_list"] = md_table_list_34183327_table_2
    state["caption"] = caption_34183327_table_2
    state["step_callback"] = step_callback
    state["md_table_aligned"] = md_table_aligned_34183327_table_2
    state["md_table_patient"] = md_table_patient_34183327_table_2

    step.execute(state)

    assert state["patient_list"] is not None
    assert type(state["patient_list"]) == list

def test_PatientMatchingAgentStep_18426260_table_0(
    llm,
    md_table_patient_refined_18426260_table_0,
    md_table_list_18426260_table_0,
    caption_18426260_table_0,
    md_table_aligned_18426260_table_0,
    md_table_patient_18426260_table_0,
    step_callback,
):
    step = PatientMatchingAgentStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_patient_refined"] = md_table_patient_refined_18426260_table_0
    state["md_table_list"] = md_table_list_18426260_table_0
    state["caption"] = caption_18426260_table_0
    state["step_callback"] = step_callback
    state["md_table_aligned"] = md_table_aligned_18426260_table_0
    state["md_table_patient"] = md_table_patient_18426260_table_0

    step.execute(state)

    assert state["patient_list"] is not None
    assert type(state["patient_list"]) == list