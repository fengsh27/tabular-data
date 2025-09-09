from collections.abc import Callable
from typing import Optional, TypedDict
from langchain_openai.chat_models.base import BaseChatOpenAI
import pandas as pd

from extractor.agents.common_agent.common_agent_2steps import CommonAgentTwoSteps
from extractor.agents.common_agent.common_agent import CommonAgent


class PKSumWorkflowState(TypedDict):
    """state data"""

    llm: BaseChatOpenAI
    md_table: str
    caption: str
    title: Optional[str] # paper title
    col_mapping: Optional[dict]
    md_table_drug: Optional[str]
    md_table_patient: Optional[str]
    md_table_patient_refined: Optional[str]
    md_table_summary: Optional[str]
    md_table_aligned: Optional[str]
    md_table_list: Optional[list[str]]
    type_unit_list: Optional[list[str]]
    drug_list: Optional[list[str]]
    patient_list: Optional[list[str]]
    value_list: Optional[list[str]]  # value table list
    df_combined: Optional[pd.DataFrame]
    previous_errors: Optional[str]

    step_callback: Optional[Callable]  # StepCallback


def pk_sum_enter_step(
    state: PKSumWorkflowState, step_name: str, step_description: Optional[str] = None
):
    if "step_callback" not in state or state["step_callback"] is None:
        return
    state["step_callback"](step_name=step_name, step_description=step_description)


def pk_sum_leave_step(
    state: PKSumWorkflowState,
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

def get_common_agent(llm: BaseChatOpenAI):
    # return CommonAgentTwoSteps(llm=llm)
    return CommonAgent(llm=llm)