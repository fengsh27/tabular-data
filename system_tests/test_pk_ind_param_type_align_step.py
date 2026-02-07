import pytest

from extractor.agents.pk_individual.pk_ind_param_type_align_step import ParametertypeAlignStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

def test_ParametertypeAlignStep_29100749_table_2(
    llm,
    md_table_individual_29100749_table_2,
    step_callback,
):
    step = ParametertypeAlignStep()
    state = PKIndWorkflowState()
    state["llm"] = llm
    state["md_table_individual"] = md_table_individual_29100749_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["md_table_aligned"] is not None
    assert type(state["md_table_aligned"]) == str