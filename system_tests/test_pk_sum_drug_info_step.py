
import pytest
from extractor.agents.pk_summary.pk_sum_drug_info_step import DrugInfoExtractionStep
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

@pytest.mark.skip()
def test_DrugInfoExtractionStep_16143486_table_4(
    llm, md_table_16143486_table_4, caption_16143486_table_4, step_callback
):
    step = DrugInfoExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table"] = md_table_16143486_table_4
    state["caption"] = caption_16143486_table_4
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["md_table_drug"] is not None
    assert type(state["md_table_drug"]) == str

def test_DrugInfoExtractionStep_22050870_table_3(
    llm,
    md_table_22050870_table_3, 
    caption_22050870_table_3, 
    title_22050870,
    step_callback,
):
    step = DrugInfoExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table"] = md_table_22050870_table_3
    state["caption"] = caption_22050870_table_3
    state["title"] = title_22050870
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["md_table_drug"] is not None
    assert type(state["md_table_drug"]) == str

