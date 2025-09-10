import asyncio
from typing import Callable, Optional, Awaitable
from langchain_openai.chat_models.base import BaseChatOpenAI

import logging

from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, extract_pmid_info_to_db, increase_token_usage

from extractor.agents.pk_pe_agents.pk_pe_design_step import PKPEDesignStep
from extractor.agents_manager.pe_study_task import (
    PEStudyInfoTask,
    PEStudyOutcomeTask,
)
from extractor.agents_manager.pk_fulltext_tool_task import (
    PKDrugIndividualTask,
    PKDrugSummaryTask,
    PKSpecimenIndividualTask,
    PKSpecimenSummaryTask,
)
from extractor.agents_manager.pk_individual_task import PKIndividualTask
from extractor.agents_manager.pk_pe_agenttool_task import PKPEAgentToolTask
from extractor.agents_manager.pk_populattion_task import PKPopulationIndividualTask, PKPopulationSummaryTask
from extractor.agents_manager.pk_summary_task import PKSummaryTask
from extractor.constants import PipelineTypeEnum
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
        pmid, _, _, _, _, _ = extract_pmid_info_to_db(pmid, pmid_db)
        return pmid is not None

    def _identification_step(self, pmid: str) -> PKPECurationWorkflowState:
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
        if state["paper_type"] == PaperTypeEnum.Neither:
            return state
        design_step = PKPEDesignStep(llm=self.llm)
        state = design_step.execute(state)
        return state

    def _curating_start_job(self, pmid: str, job_name: str,curation_callback: Optional[Callable] = None):
        if curation_callback is None:
            return
        curation_callback(pmid, job_name)
    
    def _curating_end_job(self, pmid: str, job_name: str, result: PKPECuratedTables, curation_callback: Optional[Callable] = None):
        if curation_callback is None:
            return
        curation_callback(pmid, job_name, result)

    async def _curating_start_job_async(
        self, 
        pmid: str, 
        job_name: str, 
        curation_callback: Awaitable[Callable[[str, str], None]] = None
    ):
        if curation_callback is None:
            return
        await curation_callback(pmid, job_name)

    async def _curating_end_job_async(
        self, 
        pmid: str, 
        job_name: str, 
        result: PKPECuratedTables, 
        curation_callback: Awaitable[Callable[[str, str, PKPECuratedTables], None]] = None
    ):
        if curation_callback is None:
            return
        await curation_callback(pmid, job_name, result)

    def _get_pipeline(self, pipeline_type: PipelineTypeEnum):
        if pipeline_type == PipelineTypeEnum.PK_SUMMARY:
            return PKSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db)
        elif pipeline_type == PipelineTypeEnum.PK_INDIVIDUAL:
            return PKIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db)
        elif pipeline_type == PipelineTypeEnum.PK_SPEC_SUMMARY:
            return PKSpecimenSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db)
        elif pipeline_type == PipelineTypeEnum.PK_DRUG_SUMMARY:
            return PKDrugSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db)
        elif pipeline_type == PipelineTypeEnum.PK_POPU_SUMMARY:
            return PKPopulationSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db)
        elif pipeline_type == PipelineTypeEnum.PK_SPEC_INDIVIDUAL:
            return PKSpecimenIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db)
        elif pipeline_type == PipelineTypeEnum.PK_DRUG_INDIVIDUAL:
            return PKDrugIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db)
        elif pipeline_type == PipelineTypeEnum.PK_POPU_INDIVIDUAL:
            return PKPopulationIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db)
        elif pipeline_type == PipelineTypeEnum.PE_STUDY_INFO:
            return PEStudyInfoTask(self.llm, self.pmid_db, self.print_step)
        elif pipeline_type == PipelineTypeEnum.PE_STUDY_OUTCOME:
            return PEStudyOutcomeTask(self.llm, self.pmid_db, self.print_step)
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

    def _run_pipelines(
        self, 
        pmid: str, 
        pipelines: dict[PipelineTypeEnum, PKPEAgentToolTask],
        curation_start_callback: Optional[Callable[[str, str], None]] = None, 
        curation_end_callback: Optional[Callable[[str, str, PKPECuratedTables], None]] = None
    ):
        curated_tables = {}
        for pipeline_type, pipeline in pipelines.items():
            try:
                self._curating_start_job(pmid, pipeline_type, curation_start_callback)
                correct, curated_table, explanation, suggested_fix = pipeline.run(pmid)
                result = PKPECuratedTables(
                    correct=correct,
                    curated_table=curated_table,
                    explanation=explanation,
                    suggested_fix=suggested_fix,
                )
                self._curating_end_job(pmid, pipeline_type, result, curation_end_callback)   
            except Exception as e:
                logger.error(f"Error running pmid-{pmid} {pipeline_type} workflow: \n{e}")
                continue
            curated_tables[pipeline_type] = result
        return curated_tables

    async def _run_pipelines_async(
        self, 
        pmid: str, 
        pipelines: dict[PipelineTypeEnum, PKPEAgentToolTask],
        curation_start_callback: Awaitable[Callable[[str, str], None]] = None, 
        curation_end_callback: Awaitable[Callable[[str, str, PKPECuratedTables], None]] = None
    ):
        curated_tables = {}
        for pipeline_type, pipeline in pipelines.items():
            try:
                await self._curating_start_job_async(pmid, pipeline_type, curation_start_callback)
                correct, curated_table, explanation, suggested_fix = pipeline.run(pmid)
                result = PKPECuratedTables(
                    correct=correct,
                    curated_table=curated_table,
                    explanation=explanation,
                    suggested_fix=suggested_fix,
                )
                await self._curating_end_job_async(pmid, pipeline_type, result, curation_end_callback)   
            except Exception as e:
                logger.error(f"Error running pmid-{pmid} {pipeline_type} workflow: \n{e}")
                continue
            curated_tables[pipeline_type] = result
        return curated_tables

    def _run_pk_workflows(
        self, 
        pmid: str, 
        curation_start_callback: Optional[Callable[[str, str], None]] = None, 
        curation_end_callback: Optional[Callable[[str, str, PKPECuratedTables], None]] = None
    ):
        mgrs = {
            PipelineTypeEnum.PK_SUMMARY: PKSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_INDIVIDUAL: PKIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_SPEC_SUMMARY: PKSpecimenSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_DRUG_SUMMARY: PKDrugSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_SPEC_INDIVIDUAL: PKSpecimenIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_DRUG_INDIVIDUAL: PKDrugIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_POPU_SUMMARY: PKPopulationSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_POPU_INDIVIDUAL: PKPopulationIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
        }
        return self._run_pipelines(pmid, mgrs, curation_start_callback, curation_end_callback)

    def _run_pe_workflows(
        self, 
        pmid: str, 
        curation_start_callback: Optional[Callable[[str, str], None]] = None, 
        curation_end_callback: Optional[Callable[[str, str, PKPECuratedTables], None]] = None
    ):
        mgrs = {
            PipelineTypeEnum.PE_STUDY_INFO: PEStudyInfoTask(self.llm, self.pmid_db, self.print_step),
            PipelineTypeEnum.PE_STUDY_OUTCOME: PEStudyOutcomeTask(self.llm, self.pmid_db, self.print_step),
        }
        return self._run_pipelines(pmid, mgrs, curation_start_callback, curation_end_callback)

    def _run_pk_workflows_async(
        self, 
        pmid: str, 
        curation_start_callback: Awaitable[Callable[[str, str], None]] = None, 
        curation_end_callback: Awaitable[Callable[[str, str, PKPECuratedTables], None]] = None
    ):
        mgrs = {
            PipelineTypeEnum.PK_SUMMARY: PKSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_INDIVIDUAL: PKIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_SPEC_SUMMARY: PKSpecimenSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_DRUG_SUMMARY: PKDrugSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_SPEC_INDIVIDUAL: PKSpecimenIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_DRUG_INDIVIDUAL: PKDrugIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_POPU_SUMMARY: PKPopulationSummaryTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
            PipelineTypeEnum.PK_POPU_INDIVIDUAL: PKPopulationIndividualTask(llm=self.llm, output_callback=self.print_step, pmid_db=self.pmid_db),
        }
        return self._run_pipelines_async(pmid, mgrs, curation_start_callback, curation_end_callback)

    def _run_pe_workflows_async(
        self, 
        pmid: str, 
        curation_start_callback: Awaitable[Callable[[str, str], None]] = None, 
        curation_end_callback: Awaitable[Callable[[str, str, PKPECuratedTables], None]] = None
    ):
        mgrs = {
            PipelineTypeEnum.PE_STUDY_INFO: PEStudyInfoTask(self.llm, self.pmid_db, self.print_step),
            PipelineTypeEnum.PE_STUDY_OUTCOME: PEStudyOutcomeTask(self.llm, self.pmid_db, self.print_step),
        }
        return self._run_pipelines_async(pmid, mgrs, curation_start_callback, curation_end_callback)


    def run(
        self, 
        pmid: str, 
        curation_start_callback: Optional[Callable[[str, str], None]] = None, 
        curation_end_callback: Optional[Callable[[str, str, PKPECuratedTables], None]] = None,
        pipeline_types: Optional[list[PipelineTypeEnum]] = None
    ) -> dict[str, PKPECuratedTables]:
        self._extract_pmid_info(pmid)

        
        if pipeline_types is not None:
            pipelines = {}
            for pipeline_type in pipeline_types:
                pipeline = self._get_pipeline(pipeline_type)
                pipelines[pipeline_type] = pipeline
            
            return self._run_pipelines(
                pmid, 
                pipelines, 
                curation_start_callback, 
                curation_end_callback
            ) 

        ## 1. Identification Step
        state = self._identification_step(pmid)
        paper_type = state["paper_type"]
        if paper_type == PaperTypeEnum.Neither:
            return {}

        ## execute pk summary workflow
        pk_dict = {}
        pe_dict = {}
        pipeline_tools = state["pipeline_tools"] if "pipeline_tools" in state else None
        if pipeline_tools is not None:
            return self._run_pipelines(pmid, pipeline_tools, curation_start_callback, curation_end_callback)

        if paper_type == PaperTypeEnum.PK or paper_type == PaperTypeEnum.Both:
            pk_dict = self._run_pk_workflows(pmid, curation_start_callback, curation_end_callback) # return pk curated tables
        if paper_type == PaperTypeEnum.PE or paper_type == PaperTypeEnum.Both:
            pe_dict = self._run_pe_workflows(pmid, curation_start_callback, curation_end_callback) # return pe curated tables

        return {**pk_dict, **pe_dict}

        
    async def runAsync(
        self, 
        pmid: str, 
        curation_start_callback: Awaitable[Callable[[str, str], None]] = None, 
        curation_end_callback: Awaitable[Callable[[str, str, PKPECuratedTables], None]] = None,
        pipeline_types: Optional[list[PipelineTypeEnum]] = None
    ):
        if pipeline_types is not None:
            pipelines = {}
            for pipeline_type in pipeline_types:
                pipeline = self._get_pipeline(pipeline_type)
                pipelines[pipeline_type] = pipeline
            
            return await self._run_pipelines_async(
                pmid, 
                pipelines, 
                curation_start_callback, 
                curation_end_callback
            )
        ## 1. Identification Step
        state = self._identification_step(pmid)
        paper_type = state["paper_type"]
        if paper_type == PaperTypeEnum.Neither:
            return {}

        ## execute pk summary workflow
        pk_dict = {}
        pe_dict = {}
        pipeline_tools = state["pipeline_tools"] if "pipeline_tools" in state else None
        if pipeline_tools is not None:
            return await self._run_pipelines_async(pmid, pipeline_tools, curation_start_callback, curation_end_callback)

        if paper_type == PaperTypeEnum.PK or paper_type == PaperTypeEnum.Both:
            pk_dict = await self._run_pk_workflows_async(pmid, curation_start_callback, curation_end_callback) # return pk curated tables
        if paper_type == PaperTypeEnum.PE or paper_type == PaperTypeEnum.Both:
            pe_dict = await self._run_pe_workflows_async(pmid, curation_start_callback, curation_end_callback) # return pe curated tables

        return {**pk_dict, **pe_dict}
        
