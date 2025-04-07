import pytest

from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState
from extractor.agents.pk_summary.pk_sum_patient_info_refine_step import (
    PatientInfoRefinementStep,
)


@pytest.mark.skip()
def test_PatientInfoRefinementStep(llm, md_table, caption, md_table_patient):
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption
    state["md_table"] = md_table
    state["md_table_patient"] = md_table_patient
    step = PatientInfoRefinementStep()
    step.execute(state)

    assert state["md_table_patient_refined"] is not None
    assert type(state["md_table_patient_refined"]) == str


def test_PatientInfoRefinementStep_34114632_table_3(
    llm,
    md_table_34114632_table_3,
    caption_34114632_table_3,
    md_table_patient_34114632_table_3,
):
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption_34114632_table_3
    state["md_table"] = md_table_34114632_table_3
    state["md_table_patient"] = md_table_patient_34114632_table_3
    step = PatientInfoRefinementStep()
    step.execute(state)

    assert state["md_table_patient_refined"] is not None
    assert type(state["md_table_patient_refined"]) == str
