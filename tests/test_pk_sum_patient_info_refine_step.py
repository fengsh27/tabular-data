
import pytest

from TabFuncFlow.utils.table_utils import single_html_table_to_markdown
from extractor.agents.pk_summary.pk_sum_workflow import (
    PKSumWorkflowState
)
from extractor.agents.pk_summary.pk_sum_patient_info_refine_step import (
    PatientInfoRefinementStep,
)

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


