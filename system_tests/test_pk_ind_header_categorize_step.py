import pytest

from extractor.agents.pk_individual.pk_ind_header_categorize_step import HeaderCategorizeStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

def test_HeaderCategorizeStep_29100749_table_2(
    llm,
    md_table_aligned_29100749_table_2,
    step_callback,
):
    step = HeaderCategorizeStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["md_table_aligned"] = md_table_aligned_29100749_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["col_mapping"] is not None
    assert type(state["col_mapping"]) == dict