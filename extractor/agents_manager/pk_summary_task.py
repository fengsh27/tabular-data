from typing import Callable, Optional
from langchain_openai.chat_models.base import BaseChatOpenAI
import json
from langgraph.graph import END, START, StateGraph
import pandas as pd
import logging

from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_pe_agents.pk_pe_execution_step import PKPEExecutionStep
from extractor.agents.pk_pe_agents.pk_pe_verification_step import PKPECuratedTablesVerificationStep
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow
from extractor.constants import MAX_STEP_COUNT
from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.utils import convert_html_to_text_no_table, remove_references
from extractor.agents.pk_pe_agents.pk_pe_identification_step import PKPEIdentificationStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState, PaperTypeEnum
from extractor.agents.pk_pe_agents.pk_pe_agent_tools import (
    PKSummaryTablesCurationTool,
)
from .pk_pe_agenttool_task import PKPEAgentToolTask

logger = logging.getLogger(__name__)

class PKSummaryTask(PKPEAgentToolTask):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        pmid_db: PMIDDB | None = None,
        output_callback: Callable | None = None,
    ):
        super().__init__(llm, pmid_db, output_callback)

    def _create_tool(self, pmid: str):
        return PKSummaryTablesCurationTool(
            pmid=pmid,
            llm=self.llm,
            output_callback=self.print_step,
            pmid_db=self.pmid_db,
        )

    def _get_domain(self) -> str:
        return "pharmacokinetic summary"

    def _get_paper_type(self) -> PaperTypeEnum:
        return PaperTypeEnum.PK
        
