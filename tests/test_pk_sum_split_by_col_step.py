import pytest

from extractor.agents.pk_summary.pk_sum_split_by_col_agent import SplitByColumnsResult
from extractor.agents.pk_summary.pk_sum_split_by_col_step import (
    SplitByColumnsStep
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

md_table_aligned = """
| Parameter type | N | Range | Mean ± s.d. | Median |
| --- | --- | --- | --- | --- |
| Cmax(ng/mL) | 15 | 29.3–209.6 | 56.1 ± 44.9 | 42.2 |
| AUC0−∞ | 15 | 253.3–3202.5 | 822.5 ± 706.1 | 601.5 |
| CL(mL/min/kg) | 15 | 3.33–131.50 | 49.33 ± 30.83 | 41.50 |
| CL(mL/min/m) | 15 | 5.5–67.5 | 31.95 ± 13.99 | 32.34 |
| Vdz(L/kg) | 15 | 0.33–4.05 | 1.92 ± 0.84 | 1.94 |
| T1/2(hr) | 15 | 9.5–47.0 | 20.5 ± 10.2 | 18.1 |
"""

col_mapping = {
    "Parameter type": "Parameter type",
    "N": "Uncategorized",
    "Range": "Parameter value",
    "Mean ± s.d.": "Parameter value",
    "Median": "Parameter value"
}

def test_SplitByColumnsStep(llm):
    # res = SplitByColumnsResult(re)
    step = SplitByColumnsStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state["col_mapping"] = col_mapping
    state["md_table_aligned"] = md_table_aligned

    step.execute(state)

    assert state["md_table_list"] is not None
    assert type(state["md_table_list"]) == list