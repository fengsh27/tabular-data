import pytest

from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState
from extractor.agents.pk_summary.pk_sum_patient_info_step import (
    PatientInfoExtractionStep,
)

@pytest.mark.skip()
def test_PatientInfoRefinementStep_22050870_table_3(
    llm, 
    md_table_22050870_table_3,
    caption_22050870_table_3, 
):
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption_22050870_table_3
    state["md_table"] = md_table_22050870_table_3
    step = PatientInfoExtractionStep()
    step.execute(state)

    assert state["md_table_patient_refined"] is not None
    assert type(state["md_table_patient_refined"]) == str

@pytest.mark.skip()
def test_PatientInfoRefinementStep_34183327_table_2(
    llm, 
    md_table_34183327_table_2,
    caption_34183327_table_2, 
):
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption_34183327_table_2
    state["md_table"] = md_table_34183327_table_2
    step = PatientInfoExtractionStep()
    step.execute(state)

    assert state["md_table_patient_refined"] is not None
    assert type(state["md_table_patient_refined"]) == str

def test_PatientInfoRefinementStep_34114632_table_2(
    llm, 
    step_callback,
    md_table_34114632_table_2,
    caption_34114632_table_2, 
    md_table_drug_34114632_table_2,
    title_34114632_table_2,
):
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption_34114632_table_2
    state["md_table"] = md_table_34114632_table_2
    state["step_callback"] = step_callback
    state["md_table_drug"] = md_table_drug_34114632_table_2
    state["title"] = title_34114632_table_2
    step = PatientInfoExtractionStep()
    step.execute(state)

    assert state["md_table_patient"] is not None
    assert type(state["md_table_patient"]) == str
