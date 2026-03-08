
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

@pytest.mark.skip(reason="Skipping for now")
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

@pytest.mark.skip()
def test_ParametertypeAlignStep_19925470_table_3(
    llm,
    md_table_summary_19925470_table_3,
):
    step = ParametertypeAlignStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["md_table_summary"] = md_table_summary_19925470_table_3

    state = step.execute(state)

    assert state is not None

def test_ParametertypeAlignStep_20071999_table_3(
    llm,
    step_callback,
):
    md_table_summary = """
| Dose(# of Subjects) | 5 mg(n=28) | 10 mg(n=29) | 15 mg(n=29) |
| --- | --- | --- | --- |
| Cmin (ng/ml) | 42 (CV 52%)(Range 16 – 101) | 91 (CV 76%)(Range 32 – 416) | 121 (CV 43%)(Range 19 – 204) |
| Cmax (ng/ml) | 63 (CV 36%)(Range 35 – 122) | 131 (CV 54%)(Range 66 – 459) | 181 (CV 30%)(Range 64 – 266) |
| AUC (hr*ng/mL) | 1172 (CV 46%)(Range 524 – 2582) | 2502 (CV 67%)(Range 1041 – 10286) | 3390 (CV 38%)(Range 731 – 5472) |
"""
    step = ParametertypeAlignStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["step_callback"] = step_callback
    state["md_table_summary"] = md_table_summary

    state = step.execute(state)

    assert state is not None

