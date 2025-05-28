from collections.abc import Callable
from typing import Optional, TypedDict
from langchain_openai.chat_models.base import BaseChatOpenAI
import pandas as pd


class PKSpecIndWorkflowState(TypedDict):
    """state data"""

    llm: BaseChatOpenAI
    full_text: str  # paper full text
    title: Optional[str]  # paper title
    md_table_specimen: Optional[str]
    md_table_patient_refined: Optional[str]
    md_table_time: Optional[str]
    df_combined: Optional[pd.DataFrame]
    step_callback: Optional[Callable]  # StepCallback


def pk_spec_ind_enter_step(
    state: PKSpecIndWorkflowState, step_name: str, step_description: Optional[str] = None
):
    if "step_callback" not in state or state["step_callback"] is None:
        return
    state["step_callback"](step_name=step_name, step_description=step_description)


def pk_spec_ind_leave_step(
    state: PKSpecIndWorkflowState,
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
