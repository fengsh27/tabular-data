from extractor.agents.pk_summary.pk_sum_drug_matching_step import (
    DrugMatchingAutomaticStep,
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState


def test_DrugMatchingAutomaticStep_30825333_table_2(
    llm,
    md_table_drug_30825333_table_2,
    md_table_list_30825333_table_2,
    caption_30825333_table_2,
    step_callback,
):
    step = DrugMatchingAutomaticStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_drug"] = md_table_drug_30825333_table_2
    state["md_table_list"] = md_table_list_30825333_table_2
    state["caption"] = caption_30825333_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["drug_list"] is not None
    assert type(state["drug_list"]) == list
