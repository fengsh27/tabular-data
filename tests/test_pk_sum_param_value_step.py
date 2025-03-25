import logging
from typing import Optional
from extractor.agents.pk_summary.pk_sum_param_value_step import (
    ParameterValueExtractionStep
)
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

logger = logging.getLogger(__name__)

def test_ParamterValueExtractionStep(
    llm, caption, md_table_aligned, md_table_list, step_callback
):
    step = ParameterValueExtractionStep()
    state = PKSumWorkflowState()
    state['llm'] = llm
    state['caption'] = caption
    state['md_table_aligned'] = md_table_aligned
    state['md_table_list'] = md_table_list
    state['step_callback'] = step_callback

    updated_state = step.execute(state)

    assert updated_state['value_list'] is not None
    assert type(updated_state['value_list']) == list
    

