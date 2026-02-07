import logging

from langchain_openai.chat_models.base import BaseChatOpenAI

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE
from extractor.agents.common_agent.common_step import CommonStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState


from .pk_pe_agent_tools import (
    AgentTool,
)

logger = logging.getLogger(__name__)

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
        md_curated_table = dataframe_to_markdown(df)
        logger.info(f"Curated table: \n{md_curated_table}")
        state["curated_table"] = md_curated_table
        state["source_tables"] = source_tables
        return state, {**DEFAULT_TOKEN_USAGE}




