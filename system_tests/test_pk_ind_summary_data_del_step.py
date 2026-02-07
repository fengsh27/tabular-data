import pytest
from extractor.agents.pk_individual.pk_ind_summary_data_del_step import SummaryDataDelStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

def test_SummaryDataDelStep_18426260_table_0(llm, md_table_18426260_table_0, caption_18426260_table_0, step_callback):
    step = SummaryDataDelStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["md_table"] = md_table_18426260_table_0
    state["caption"] = caption_18426260_table_0
    state["step_callback"] = step_callback
    step.execute(state)
    assert state["md_table_individual"] is not None