

from typing import Optional, TypedDict
from langchain_openai.chat_models.base import BaseChatOpenAI

from extractor.agents.agent_utils import StepCallback

class PKSumWorkflowState(TypedDict):
    """ state data """
    llm: BaseChatOpenAI
    md_table: str
    caption: str
    col_mapping: Optional[dict] = None
    md_table_drug: Optional[str] = None
    md_table_patient: Optional[str] = None
    md_table_patient_refined: Optional[str] = None
    md_table_summary: Optional[str] = None
    md_table_aligned: Optional[str] = None
    md_table_list: Optional[list[str]] = None
    type_unit_list: Optional[list[str]] = None
    step_callback: Optional[StepCallback] = None
  

def pk_sum_enter_step(
    state: PKSumWorkflowState, 
    step_name: str, 
    step_description: Optional[str]=None
):
    if not "step_callback" in state or state["step_callback"] is None:
        return
    state["step_callback"](
        step_name=step_name,
        step_description=step_description
    )
def pk_sum_leave_step(
    state: PKSumWorkflowState, 
    step_output: str, 
    step_reasoning_process: str,
    token_usage: dict
):
    if not "step_callback" in state or state["step_callback"] is None:
        return
    state["step_callback"](
        step_output=step_output,
        step_reasoning_process=step_reasoning_process,
        token_usage=token_usage,
    )

