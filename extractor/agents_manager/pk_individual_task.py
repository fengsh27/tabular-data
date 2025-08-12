from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI

import logging

from extractor.database.pmid_db import PMIDDB

from extractor.agents.pk_pe_agents.pk_pe_agents_types import PaperTypeEnum
from extractor.agents.pk_pe_agents.pk_pe_agent_tools import (
    PKIndividualTablesCurationTool,
)
from .pk_pe_agenttool_task import PKPEAgentToolTask

logger = logging.getLogger(__name__)

class PKIndividualTask(PKPEAgentToolTask):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        pmid_db: PMIDDB | None = None,
        output_callback: Callable | None = None,
    ):
        super().__init__(llm, pmid_db, output_callback)
        self.task_name = "PK Individual Task"

    def _create_tool(self, pmid: str):
        return PKIndividualTablesCurationTool(
            pmid=pmid,
            llm=self.llm,
            output_callback=self.print_step,
            pmid_db=self.pmid_db,
        )

    def _get_domain(self) -> str:
        return "pharmacokinetic population and individual"

    def _get_paper_type(self) -> PaperTypeEnum:
        return PaperTypeEnum.PK
        
