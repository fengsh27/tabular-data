import pytest

from extractor.agents.pk_individual.pk_ind_drug_info_step import DrugInfoExtractionStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

def test_PKIndDrugInfoStep_29100749_table_0(
    llm,
    source_table_29100749_table_0,
    caption_29100749_table_0,
    title_29100749,
    abstract_29100749,
    step_callback,
):
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["md_table"] = source_table_29100749_table_0
    state["caption"] = caption_29100749_table_0
    state["title"] = title_29100749
    state["abstract"] = abstract_29100749
    state["step_callback"] = step_callback
    step = DrugInfoExtractionStep()
    step.execute(state)

    assert state["md_table_drug"] is not None
    assert type(state["md_table_drug"]) == str