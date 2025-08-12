
from langchain_openai.chat_models.base import BaseChatOpenAI

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE
from extractor.agents.common_agent.common_step import CommonStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState
from extractor.database.pmid_db import PMIDDB

from .pk_pe_agent_tools import (
    AgentTool,
)

class PKPEExecutionStep(CommonStep):
    def __init__(
        self, 
        llm: BaseChatOpenAI,
        tool: AgentTool,
    ):
        super().__init__(llm)
        self.step_name = "PK PE Execution Step"
        self.tool = tool

    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        state: PKPECurationWorkflowState = state
        previous_errors = state["previous_errors"] if "previous_errors" in state else None
        previous_errors = previous_errors if previous_errors is not None else "N/A"
        df, source_tables = self.tool.run(previous_errors)
        if df is None:
            return state, {**DEFAULT_TOKEN_USAGE}
        state["curated_table"] = dataframe_to_markdown(df)
        state["source_tables"] = source_tables
        return state, {**DEFAULT_TOKEN_USAGE}




