
import pytest

from extractor.agents.pk_summary.pk_sum_param_type_align_step import (
    ParametertypeAlignStep
)
from extractor.agents.pk_summary.pk_sum_param_type_align_agent import (
    ParameterTypeAlignResult
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import (
    PKSumWorkflowState
)

def test_ParametertypeAlignStep_35465728_table_2(
    llm,
    md_table_summary_35465728_table_2,
):
    the_obj = ParameterTypeAlignResult.model_json_schema()
    step = ParametertypeAlignStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_summary"] = md_table_summary_35465728_table_2

    state = step.execute(state)

    assert state is not None

