from typing import Optional
from langchain_openai.chat_models.base import BaseChatOpenAI

import logging

from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage

from extractor.agents_manager.pe_study_manager import (
    PEStudyInfoManager,
    PEStudyOutcomeManager,
)
from extractor.agents_manager.pk_fulltext_tool_manager import (
    PKDrugIndividualManager,
    PKDrugSummaryManager,
    PKSpecimenIndividualManager,
    PKSpecimenSummaryManager,
)
from extractor.agents_manager.pk_individual_manager import PKIndividualManager
from extractor.agents_manager.pk_populattion_manager import PKPopulationIndividualManager, PKPopulationSummaryManager
from extractor.agents_manager.pk_summary_manager import PKSummaryManager
from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.utils import convert_html_to_text_no_table, remove_references
from extractor.agents.pk_pe_agents.pk_pe_identification_step import PKPEIdentificationStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECuratedTables, PKPECurationWorkflowState, PaperTypeEnum


logger = logging.getLogger(__name__)

class PKPEManager:
    def __init__(self, llm: BaseChatOpenAI, pmid_db: PMIDDB | None = None):
        self.llm = llm
        self.pmid_db = pmid_db if pmid_db is not None else PMIDDB()
        self.total_token_usage = {**DEFAULT_TOKEN_USAGE}
    
    def print_step(
        self,
        step_name: Optional[str] = None,
        step_description: Optional[str] = None,
        step_output: Optional[str] = None,
        step_reasoning_process: Optional[str | list[str]] = None,
        token_usage: Optional[dict] = None,
    ):
        if step_name is not None:
            logger.info("=" * 64)
            logger.info(step_name)
        if step_description is not None:
            logger.info(step_description)
        if token_usage is not None:
            logger.info(
                f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
            )
            self.total_token_usage = increase_token_usage(self.total_token_usage, token_usage)
            logger.info(
                f"overall total tokens: {self.total_token_usage['total_tokens']}, overall prompt tokens: {self.total_token_usage['prompt_tokens']}, overall completion tokens: {self.total_token_usage['completion_tokens']}"
            )
        if step_reasoning_process is not None:
            logger.info(f"\n\n{step_reasoning_process}\n\n")
        if step_output is not None:
            logger.info(step_output)

    def _extract_pmid_info(self, pmid: str):
        pmid_db = self.pmid_db
        pmid_info = pmid_db.select_pmid_info(pmid)
        if pmid_info is not None: # pmid has been saved to database
            return True
        retriever = ArticleRetriever()
        res, html_content, code = retriever.request_article(pmid)
        if not res:
            return False
        extractor = HtmlTableExtractor()
        tables = extractor.extract_tables(html_content)
        sections = extractor.extract_sections(html_content)
        abstract = extractor.extract_abstract(html_content)
        title = extractor.extract_title(html_content)
        full_text = convert_html_to_text_no_table(html_content)
        full_text = remove_references(full_text)
        pmid_db.insert_pmid_info(pmid, title, abstract, full_text, tables, sections)
        return True

    def _identification_step(self, pmid: str):
        pmid_db = self.pmid_db
        pmid_info = pmid_db.select_pmid_info(pmid)
        state = PKPECurationWorkflowState(
            pmid=pmid,
            paper_title=pmid_info[1],
            paper_abstract=pmid_info[2],
            full_text=pmid_info[3],
            step_output_callback=self.print_step,
        )
        
        identification_step = PKPEIdentificationStep(llm=self.llm)
        state = identification_step.execute(state)
        return state

    def _run_pk_workflows(self, pmid: str):
        mgrs = {
            # "pk_summary": PKSummaryManager(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            # "pk_individual": PKIndividualManager(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            "pk_specimen_summary": PKSpecimenSummaryManager(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            "pk_drug_summary": PKDrugSummaryManager(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            "pk_specimen_individual": PKSpecimenIndividualManager(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            "pk_drug_individual": PKDrugIndividualManager(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            "pk_population_summary": PKPopulationSummaryManager(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            "pk_population_individual": PKPopulationIndividualManager(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
        }
        curated_tables = {}
        for mgr_name, mgr in mgrs.items():
            try:
                correct, curated_table, explanation, suggested_fix = mgr.run(pmid)
            except Exception as e:
                logger.error(f"Error running pmid-{pmid} {mgr_name} workflow: \n{e}")
                continue
            curated_tables[mgr_name] = PKPECuratedTables(
                correct=correct,
                curated_table=curated_table,
                explanation=explanation,
                suggested_fix=suggested_fix,
            )
        return curated_tables

    def _run_pe_workflows(self, pmid: str):
        mgrs = {
            "pe_study_info": PEStudyInfoManager(self.llm, self.pmid_db, self.print_step),
            "pe_study_output": PEStudyOutcomeManager(self.llm, self.pmid_db, self.print_step),
        }
        curated_tables = {}
        for mgr_name, mgr in mgrs.items():
            correct, curated_table, explanation, suggested_fix = mgr.run(pmid)
            curated_tables[mgr_name] = PKPECuratedTables(
                correct=correct,
                curated_table=curated_table,
                explanation=explanation,
                suggested_fix=suggested_fix,
            )
        return curated_tables

    def run(self, pmid: str) -> dict[str, PKPECuratedTables]:
        self._extract_pmid_info(pmid)

        ## 1. Identification Step
        state = self._identification_step(pmid)
        paper_type = state["paper_type"]

        ## execute pk summary workflow
        if paper_type == PaperTypeEnum.PK:
            return self._run_pk_workflows(pmid) # return pk curated tables
        elif paper_type == PaperTypeEnum.PE:
            return self._run_pe_workflows(pmid) # return pe curated tables
        else:
            raise ValueError(f"Invalid paper type: {paper_type}")

        
