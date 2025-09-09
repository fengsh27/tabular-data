from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI

import logging

from databases.pmid_db import PMIDDB

from extractor.agents.pk_pe_agents.pk_pe_agents_types import PaperTypeEnum
from extractor.agents.pk_drug_summary.pk_drug_sum_workflow import PKDrugSumWorkflow
from extractor.agents.pk_specimen_summary.pk_spec_sum_workflow import PKSpecSumWorkflow
from extractor.agents.pk_specimen_individual.pk_spec_ind_workflow import PKSpecIndWorkflow
from extractor.agents.pk_drug_individual.pk_drug_ind_workflow import PKDrugIndWorkflow
from extractor.agents.pk_pe_agents.pk_pe_agent_tools import (
    FullTextCurationTool,
)
from .pk_pe_agenttool_task import PKPEAgentToolTask

logger = logging.getLogger(__name__)

class PKSpecimenSummaryTask(PKPEAgentToolTask):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        pmid_db: PMIDDB | None = None,
        output_callback: Callable | None = None,
    ):
        super().__init__(llm, pmid_db, output_callback)
        self.task_name = "PK Specimen Summary Task"

    def _create_tool(self, pmid: str):
        return FullTextCurationTool(
            pmid=pmid,
            cls=PKSpecSumWorkflow,
            tool_name="PK Specimen Summary Curation Tool",
            tool_description="This tool is used to extract the specimen summary data from the source paper.",
            llm=self.llm,
            output_callback=self.print_step,
            pmid_db=self.pmid_db,
        )

    def _get_domain(self) -> str:
        return "pharmacokinetic specimen summary"

    def _get_paper_type(self) -> PaperTypeEnum:
        return PaperTypeEnum.PK
        
class PKDrugSummaryTask(PKPEAgentToolTask):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        pmid_db: PMIDDB | None = None,
        output_callback: Callable | None = None,
    ):
        super().__init__(llm, pmid_db, output_callback)
        self.task_name = "PK Drug Summary Task"
        
    def _create_tool(self, pmid: str):
        return FullTextCurationTool(
            pmid=pmid,
            cls=PKDrugSumWorkflow,
            tool_name="PK Drug Summary Curation Tool",
            tool_description="This tool is used to extract the drug summary data from the source paper.",
            llm=self.llm,
            output_callback=self.print_step,
            pmid_db=self.pmid_db,
        )
        
    def _get_domain(self) -> str:
        return "pharmacokinetic drug summary"

    def _get_paper_type(self) -> PaperTypeEnum:
        return PaperTypeEnum.PK

class PKSpecimenIndividualTask(PKPEAgentToolTask):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        pmid_db: PMIDDB | None = None,
        output_callback: Callable | None = None,
    ):
        super().__init__(llm, pmid_db, output_callback)
        self.task_name = "PK Specimen Individual Task"
        
    def _create_tool(self, pmid: str):
        return FullTextCurationTool(
            pmid=pmid,
            cls=PKSpecIndWorkflow,
            tool_name="PK Specimen Individual Curation Tool",
            tool_description="This tool is used to extract the specimen individual data from the source paper.",
            llm=self.llm,
            output_callback=self.print_step,
            pmid_db=self.pmid_db,
        )

    def _get_domain(self) -> str:
        return "pharmacokinetic specimen and individual"

    def _get_paper_type(self) -> PaperTypeEnum:
        return PaperTypeEnum.PK

class PKDrugIndividualTask(PKPEAgentToolTask):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        pmid_db: PMIDDB | None = None,
        output_callback: Callable | None = None,
    ):
        super().__init__(llm, pmid_db, output_callback)
        self.task_name = "PK Drug Individual Task"
        
    def _create_tool(self, pmid: str):
        return FullTextCurationTool(
            pmid=pmid,
            cls=PKDrugIndWorkflow,
            tool_name="PK Drug Individual Curation Tool",
            tool_description="This tool is used to extract the drug individual data from the source paper.",
            llm=self.llm,
            output_callback=self.print_step,
            pmid_db=self.pmid_db,
        )
        
    def _get_domain(self) -> str:
        return "pharmacokinetic drug and individual"

    def _get_paper_type(self) -> PaperTypeEnum:
        return PaperTypeEnum.PK
