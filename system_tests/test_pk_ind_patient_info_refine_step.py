
import pytest

from extractor.agents.pk_individual.pk_ind_patient_info_refine_step import (
    PatientInfoRefinementStep,
    PKIndWorkflowState
)

def test_PatientInfoRefinementStep_18426260_table_0(
    llm,
    caption_18426260_table_0,
    md_table_18426260_table_0,
    pk_ind_md_table_patient_18426260_table_0,
    step_callback,
):
    step = PatientInfoRefinementStep()
    state = PKIndWorkflowState(
        llm=llm,
        caption=caption_18426260_table_0,
        md_table=md_table_18426260_table_0,
        md_table_patient=pk_ind_md_table_patient_18426260_table_0,
        step_callback=step_callback,
    )

    state = step.execute(state)

    assert state["md_table_patient_refined"] is not None
    assert type(state["md_table_patient_refined"]) == str


