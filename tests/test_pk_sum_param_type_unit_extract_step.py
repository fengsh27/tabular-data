
from extractor.agents.pk_summary.pk_sum_param_type_unit_extract_step import (
    ExtractParamTypeAndUnitStep,
    ParamTypeUnitExtractionResult,
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

def test_ExtractParamTypeAndUnitStep(
    llm, 
    col_mapping,
    md_table_aligned, 
    caption, 
    md_table_list
):
    step = ExtractParamTypeAndUnitStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['col_mapping'] = col_mapping
    state['md_table_aligned'] = md_table_aligned
    state['md_table_list'] = md_table_list
    state['caption'] = caption

    step.execute(state)

    assert state["type_unit_list"] is not None
    assert type(state["type_unit_list"]) == list