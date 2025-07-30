from abc import ABC, abstractmethod
from typing import Any, Callable
from langchain_openai.chat_models.base import BaseChatOpenAI
import pandas as pd
import logging

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pe_study_outcome_ver2.pe_study_out_workflow import PEStudyOutWorkflow
from extractor.agents.pk_individual.pk_ind_workflow import PKIndWorkflow
from extractor.agents.pk_population_summary.pk_popu_sum_workflow import PKPopuSumWorkflow
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow, PKSumWorkflowState
from extractor.agents.pk_population_individual.pk_popu_ind_workflow import PKPopuIndWorkflow
from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.table_utils import select_pe_tables, select_pk_demographic_tables, select_pk_summary_tables
from extractor.utils import convert_html_to_text_no_table, remove_references

logger = logging.getLogger(__name__)

class AgentTool(ABC):
    def __init__(
        self,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
    ):
        self.llm = llm
        self.output_callback = output_callback

    def _print_token_usage(self, token_usage: dict):
        if self.output_callback is not None:
            self.output_callback(token_usage=token_usage)
    def _print_step_output(self, step_output: str):
        if self.output_callback is not None:
            self.output_callback(step_output=step_output)

    def _print_tool_name(self):
        if self.output_callback is not None:
            self.output_callback(step_name=self.__class__.__name__)

    @abstractmethod
    def _run(self):
        pass

    def run(self):
        self._print_tool_name()
        return self._run()

class PKSummaryTablesCurationTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid

    def _run(self):
        pmid_db = PMIDDB()
        pmid_info = pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:
            return None
        tables = pmid_info[4]
        selected_tables, indexes, reasoning_process, token_usage = select_pk_summary_tables(tables, self.llm)
        self._print_step_output(reasoning_process)
        self._print_token_usage(token_usage)
        title = pmid_info[1]
        workflow = PKSumWorkflow(llm=self.llm)
        workflow.build()
        dfs: list[pd.DataFrame] = []
        for table in selected_tables:
            caption = "\n".join([table["caption"], table["footnote"]])
            df = workflow.go_md_table(
                title=title,
                md_table=dataframe_to_markdown(table["table"]),
                caption_and_footnote=caption,
            )
            dfs.append(df)

        # combine dfs
        df_combined = (
            pd.concat(dfs, axis=0).reset_index(drop=True)
            if len(dfs) > 0
            else pd.DataFrame()
        )
        return df_combined

class PKIndividualTablesCurationTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid
        
    def _run(self):
        pmid_db = PMIDDB()
        pmid_info = pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:   
            return None
        tables = pmid_info[4]
        title = pmid_info[1]
        selected_tables, indexes, reasoning_process, token_usage = select_pk_demographic_tables(tables, self.llm)
        self._print_step_output(reasoning_process)
        self._print_token_usage(token_usage)
        if not selected_tables:
            return None
        workflow = PKIndWorkflow(llm=self.llm)
        workflow.build()
        dfs: list[pd.DataFrame] = []
        for table in selected_tables:
            caption = "\n".join([table["caption"], table["footnote"]])
            df = workflow.go_md_table(
                title=title,
                md_table=dataframe_to_markdown(table["table"]),
                caption_and_footnote=caption,
            )
            dfs.append(df)
        df_combined = (
            pd.concat(dfs, axis=0).reset_index(drop=True)
            if len(dfs) > 0
            else pd.DataFrame()
        )
        return df_combined

class PKPopulationSummaryWorkflowTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid
        
    def _run(self):
        pmid_db = PMIDDB()
        pmid_info = pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:
            return None
        title = pmid_info[1]
        sections = pmid_info[5]
        abstract = pmid_info[2]
        tables = pmid_info[4]
        selected_tables, indexes, reasoning_process, token_usage = select_pk_demographic_tables(tables, self.llm)
        self._print_step_output(reasoning_process)
        self._print_token_usage(token_usage)
        
        if not selected_tables:
            logger.info("No PK demographic table detected. Use full text as the input.")

            if sections:
                article_text = "\n".join(
                    f"{sec['section']}\n{sec['content']}" for sec in sections
                )
            else:
                article_text = f"{title}\n{abstract}"
            article_text = convert_html_to_text_no_table(article_text)
            article_text = remove_references(article_text)
            workflow = PKPopuSumWorkflow(llm=self.llm)
            workflow.build()
            result_df = workflow.go_full_text(
                title=title,
                full_text=article_text,
            )
        else:
            logger.info("Detected PK demographic table. Use the table as the input.")
            workflow = PKPopuIndWorkflow(llm=self.llm)
            workflow.build()
            dfs: list[pd.DataFrame] = []
            for table in selected_tables:
                caption = "\n".join([table["caption"], table["footnote"]])
                df = workflow.go_full_text(
                    title=title,
                    full_text=dataframe_to_markdown(table["table"])+"\n\n"+caption,
                    step_callback=self.output_callback,
                )
                dfs.append(df)
            result_df = pd.concat(dfs, ignore_index=True)
        return result_df


class PKPopulationIndividualWorkflowTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid
        
    def _run(self):
        pmid_db = PMIDDB()
        pmid_info = pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:
            return None
        title = pmid_info[1]
        sections = pmid_info[5]
        abstract = pmid_info[2]
        tables = pmid_info[4]
        selected_tables, indexes, reasoning_process, token_usage = select_pk_demographic_tables(tables, self.llm)
        self._print_step_output(reasoning_process)
        self._print_token_usage(token_usage)

        if not selected_tables:
            logger.info("No PK demographic table detected. Use full text as the input.")
            if sections:
                article_text = "\n".join(
                    f"{sec['section']}\n{sec['content']}" for sec in sections
                )
            else:
                article_text = f"{title}\n{abstract}"
            article_text = convert_html_to_text_no_table(article_text)
            article_text = remove_references(article_text)
            workflow = PKPopuIndWorkflow(llm=self.llm)
            workflow.build()
            result_df = workflow.go_full_text(
                title=title,
                full_text=article_text,
                step_callback=self.output_callback,
            )
            return result_df
        else:
            logger.info("Detected PK demographic table. Use the table as the input.")
            workflow = PKPopuIndWorkflow(llm=self.llm)
            workflow.build()
            dfs: list[pd.DataFrame] = []
            for table in selected_tables:
                caption = "\n".join([table["caption"], table["footnote"]])
                df = workflow.go_full_text(
                    title=title,
                    full_text=dataframe_to_markdown(table["table"])+"\n\n"+caption,
                    step_callback=self.output_callback,
                )
                dfs.append(df)
            result_df = pd.concat(dfs, ignore_index=True)
            return result_df

class PEStudyOutcomeWorkflowTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid
    
    def _run(self):
        pmid_db = PMIDDB()
        pmid_info = pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:
            return None
        title = pmid_info[1]
        sections = pmid_info[5]
        abstract = pmid_info[2]
        tables = pmid_info[4]
        selected_tables, indexes, reasoning_process, token_usage = select_pe_tables(tables, self.llm)
        self._print_step_output(reasoning_process)
        self._print_token_usage(token_usage)
        if not selected_tables:
            return None
        workflow = PEStudyOutWorkflow(llm=self.llm)
        workflow.build()
        dfs: list[pd.DataFrame] = []
        for table in selected_tables:
            caption = "\n".join([table["caption"], table["footnote"]])
            df = workflow.go_md_table(
                title=title,
                md_table=dataframe_to_markdown(table["table"]),
                caption_and_footnote=caption,
            )
            dfs.append(df)
        df_combined = (
            pd.concat(dfs, axis=0).reset_index(drop=True)
            if len(dfs) > 0
            else pd.DataFrame()
        )
        return df_combined

class FullTextWorkflowTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        cls: Any,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
    ):
        """
        cls: the class of the workflow to be used
        PKSpecSumWorkflow
        PKDrugSumWorkflow
        PKSpecIndWorkflow
        PKDrugIndWorkflow
        PEStudyInfoWorkflow
        """
        super().__init__(llm, output_callback)
        self.pmid = pmid
        self.cls = cls
        
    def _run(self):
        pmid_db = PMIDDB()
        pmid_info = pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:
            return None
        title = pmid_info[1]
        sections = pmid_info[5]
        abstract = pmid_info[2] 
        wf = self.cls(llm=self.llm)
        wf.build()
        if sections:
            article_text = "\n".join(
                f"{sec['section']}\n{sec['content']}" for sec in sections
            )
        else:
            article_text = f"{title}\n{abstract}"
        article_text = convert_html_to_text_no_table(article_text)
        article_text = remove_references(article_text)
        return wf.go_full_text(
            title=title,
            full_text=article_text,
            step_callback=self.output_callback,
        )