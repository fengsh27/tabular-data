from collections.abc import Callable
from typing import Optional, TypedDict
from langchain_openai.chat_models.base import BaseChatOpenAI
import pandas as pd


class PKIndWorkflowState(TypedDict):
    """state data"""

    llm: BaseChatOpenAI
    md_table: str
    caption: str
    llm2: Optional[BaseChatOpenAI]
    title: Optional[str] # paper title
    col_mapping: Optional[dict]
    md_table_drug: Optional[str]
    md_table_patient: Optional[str]
    md_table_patient_refined: Optional[str]
    md_table_individual: Optional[str]
    md_table_aligned: Optional[str]
    md_table_list: Optional[list[str]]
    type_unit_list: Optional[list[str]]
    drug_list: Optional[list[str]]
    patient_list: Optional[list[str]]
    # value_list: Optional[list[str]]  # value table list
    time_list: Optional[list[str]]
    df_combined: Optional[pd.DataFrame]
    previous_errors: Optional[str]
    full_text: Optional[str]

    step_callback: Optional[Callable]  # StepCallback


def pk_ind_enter_step(
    state: PKIndWorkflowState, step_name: str, step_description: Optional[str] = None
):
    if "step_callback" not in state or state["step_callback"] is None:
        return
    state["step_callback"](step_name=step_name, step_description=step_description)


def pk_ind_leave_step(
    state: PKIndWorkflowState,
    step_output: str,
    step_reasoning_process: str,
    token_usage: dict,
):
    if "step_callback" not in state or state["step_callback"] is None:
        return
    state["step_callback"](
        step_output=step_output,
        step_reasoning_process=step_reasoning_process,
        token_usage=token_usage,
    )
