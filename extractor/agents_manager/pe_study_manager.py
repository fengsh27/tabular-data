from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI

import logging

from extractor.agents.pe_study_info.pe_study_info_workflow import PEStudyInfoWorkflow
from extractor.database.pmid_db import PMIDDB

from extractor.agents.pk_pe_agents.pk_pe_agents_types import PaperTypeEnum
from extractor.agents.pk_pe_agents.pk_pe_agent_tools import (
    FullTextCurationTool,
    PEStudyOutcomeCurationTool,
)
from .pk_pe_agenttool_manager import PKPEAgentToolManager

logger = logging.getLogger(__name__)

class PEStudyOutcomeManager(PKPEAgentToolManager):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        pmid_db: PMIDDB | None = None,
        output_callback: Callable | None = None,
    ):
        super().__init__(llm, pmid_db, output_callback)

    def _create_tool(self, pmid: str):
        return PEStudyOutcomeCurationTool(
            pmid=pmid,
            llm=self.llm,
            output_callback=self.print_step,
            pmid_db=self.pmid_db,
        )

    def _get_domain(self) -> str:
        return "pharmacoeconomic study outcome"

    def _get_paper_type(self) -> PaperTypeEnum:
        return PaperTypeEnum.PE
        
class PEStudyInfoManager(PKPEAgentToolManager):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        pmid_db: PMIDDB | None = None,
        output_callback: Callable | None = None,
    ):
        super().__init__(llm, pmid_db, output_callback)
        
    def _create_tool(self, pmid: str):
        return FullTextCurationTool(
            pmid=pmid,
            cls=PEStudyInfoWorkflow,
            tool_name="PE Study Info Curation Tool",
            tool_description="This tool is used to extract the study information from the source paper.",
            llm=self.llm,
            output_callback=self.print_step,
            pmid_db=self.pmid_db,
        )
        
    def _get_domain(self) -> str:
        return "pharmacoeconomic study information"

    def _get_paper_type(self) -> PaperTypeEnum:
        return PaperTypeEnum.PE
