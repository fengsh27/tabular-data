import pytest

from extractor.agents.pk_individual.pk_ind_patient_info_step import PatientInfoExtractionStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

def test_PatientInfoStep_33253437_table_1(
    llm,
    step_callback,
    caption_table_1_33253437,
    md_table_individual_table_1_33253437,
):
    step = PatientInfoExtractionStep()
    state = PKIndWorkflowState(
        llm=llm,
        caption=caption_table_1_33253437,
        md_table=md_table_individual_table_1_33253437,
        step_callback=step_callback,
    )

    state = step.execute(state)

    assert state["md_table_patient"] is not None
    assert type(state["md_table_patient"]) == str
