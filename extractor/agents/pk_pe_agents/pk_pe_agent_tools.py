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
    def _get_tool_name(self) -> str:
        pass

    @abstractmethod
    def _get_tool_description(self) -> str:
        pass

    @abstractmethod
    def _run(self, previous_errors: str | None = None) -> tuple[pd.DataFrame | None, list[str] | str | None]:
        pass

    def run(self, previous_errors: str | None = None):
        self._print_tool_name()
        try:
            return self._run(previous_errors)
        except Exception as e:
            logger.error(f"Error running {self._get_tool_name()} tool: \n{e}")
            return pd.DataFrame(), "N/A"

class PKSummaryTablesCurationTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
        pmid_db: PMIDDB | None = None,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid
        self.pmid_db = pmid_db if pmid_db is not None else PMIDDB()

    def _get_tool_name(self) -> str:
        return "PK Summary Tables Curation Tool"

    def _get_tool_description(self) -> str:
        return "This tool is used to curate the PK summary tables from the source paper."

    def _run(self, previous_errors: str | None = None):
        pmid_info = self.pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:
            return None, None
        tables = pmid_info[4]
        selected_tables, indexes, reasoning_process, token_usage = select_pk_summary_tables(tables, self.llm)
        self._print_step_output(reasoning_process)
        self._print_token_usage(token_usage)
        title = pmid_info[1]
        workflow = PKSumWorkflow(llm=self.llm)
        workflow.build()
        dfs: list[pd.DataFrame] = []
        source_tables = []
        for table in selected_tables:
            caption = "\n".join([table["caption"], table["footnote"]])
            source_table = dataframe_to_markdown(table["table"])
            source_tables.append(f"caption: \n{caption}\n\n table: \n{source_table}")
            df = workflow.go_md_table(
                title=title,
                md_table=source_table,
                caption_and_footnote=caption,
                step_callback=self.output_callback,
                previous_errors=previous_errors,
            )
            dfs.append(df)

        # combine dfs
        df_combined = (
            pd.concat(dfs, axis=0).reset_index(drop=True)
            if len(dfs) > 0
            else pd.DataFrame()
        )
        return df_combined, source_tables

class PKIndividualTablesCurationTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
        pmid_db: PMIDDB | None = None,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid
        self.pmid_db = pmid_db if pmid_db is not None else PMIDDB()
    
    def _get_tool_name(self) -> str:
        return "PK Individual Tables Curation Tool"

    def _get_tool_description(self) -> str:
        return "This tool is used to extract the individual tables from the source paper."

    def _run(self, previous_errors: str):
        pmid_info = self.pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:   
            return None
        tables = pmid_info[4]
        title = pmid_info[1]
        selected_tables, indexes, reasoning_process, token_usage = select_pk_demographic_tables(tables, self.llm)
        self._print_step_output(reasoning_process)
        self._print_token_usage(token_usage)
        if not selected_tables:
            return None, None
        workflow = PKIndWorkflow(llm=self.llm)
        workflow.build()
        dfs: list[pd.DataFrame] = []
        source_tables = []
        for table in selected_tables:
            caption = "\n".join([table["caption"], table["footnote"]])
            source_table = dataframe_to_markdown(table["table"])
            source_tables.append(source_table)
            df = workflow.go_md_table(
                title=title,
                md_table=source_table,
                caption_and_footnote=caption,
                step_callback=self.output_callback,
                previous_errors=previous_errors,
            )
            dfs.append(df)
        df_combined = (
            pd.concat(dfs, axis=0).reset_index(drop=True)
            if len(dfs) > 0
            else pd.DataFrame()
        )
        return df_combined, source_tables

class PKPopulationSummaryCurationTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
        pmid_db: PMIDDB | None = None,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid
        self.pmid_db = pmid_db if pmid_db is not None else PMIDDB()
        
    def _get_tool_name(self) -> str:
        return "PK Population Summary Curation Tool"

    def _get_tool_description(self) -> str:
        return "This tool is used to extract the population summary data from the source paper."

    def _run(self, previous_errors: str):
        pmid_info = self.pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:
            return None, None
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
                step_callback=self.output_callback,
                previous_errors=previous_errors,
            )
            return result_df, article_text
        else:
            logger.info("Detected PK demographic table. Use the table as the input.")
            workflow = PKPopuIndWorkflow(llm=self.llm)
            workflow.build()
            dfs: list[pd.DataFrame] = []
            source_tables = []
            for table in selected_tables:
                caption = "\n".join([table["caption"], table["footnote"]])
                source_table = dataframe_to_markdown(table["table"])+"\n\n"+caption
                source_tables.append(source_table)
                df = workflow.go_full_text(
                    title=title,
                    full_text=source_table,
                    step_callback=self.output_callback,
                )
                dfs.append(df)
            result_df = pd.concat(dfs, ignore_index=True)
        return result_df, source_tables


class PKPopulationIndividualCurationTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
        pmid_db: PMIDDB | None = None,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid
        self.pmid_db = pmid_db if pmid_db is not None else PMIDDB()
        
    def _get_tool_name(self) -> str:
        return "PK Population Individual Curation Tool"

    def _get_tool_description(self) -> str:
        return "This tool is used to extract the population individual data from the source paper."

    def _run(self, previous_errors: str):
        pmid_info = self.pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:
            return None, None
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
                previous_errors=previous_errors,
            )
            return result_df, article_text
        else:
            logger.info("Detected PK demographic table. Use the table as the input.")
            workflow = PKPopuIndWorkflow(llm=self.llm)
            workflow.build()
            dfs: list[pd.DataFrame] = []
            source_tables = []
            for table in selected_tables:
                caption = "\n".join([table["caption"], table["footnote"]])
                source_table = dataframe_to_markdown(table["table"])+"\n\n"+caption
                source_tables.append(source_table)
                df = workflow.go_full_text(
                    title=title,
                    full_text=source_table,
                    step_callback=self.output_callback,
                    previous_errors=previous_errors,
                )
                dfs.append(df)
            result_df = pd.concat(dfs, ignore_index=True)
            return result_df, source_tables

class PEStudyOutcomeCurationTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
        pmid_db: PMIDDB | None = None,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid
        self.pmid_db = pmid_db if pmid_db is not None else PMIDDB()
    
    def _get_tool_name(self) -> str:
        return "PE Study Outcome Curation Tool"

    def _get_tool_description(self) -> str:
        return "This tool is used to extract the study outcome data from the source paper."

    def _run(self, previous_errors: str):
        pmid_info = self.pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:
            return None, None
        title = pmid_info[1]
        sections = pmid_info[5]
        abstract = pmid_info[2]
        tables = pmid_info[4]
        selected_tables, indexes, reasoning_process, token_usage = select_pe_tables(tables, self.llm)
        self._print_step_output(reasoning_process)
        self._print_token_usage(token_usage)
        if not selected_tables:
            return None, None
        workflow = PEStudyOutWorkflow(llm=self.llm)
        workflow.build()
        dfs: list[pd.DataFrame] = []
        source_tables = []
        for table in selected_tables:
            caption = "\n".join([table["caption"], table["footnote"]])
            source_table = dataframe_to_markdown(table["table"])
            source_tables.append(source_table)
            df = workflow.go_md_table(
                title=title,
                md_table=source_table,
                caption_and_footnote=caption,
                step_callback=self.output_callback,
                previous_errors=previous_errors,
            )
            dfs.append(df)
        df_combined = (
            pd.concat(dfs, axis=0).reset_index(drop=True)
            if len(dfs) > 0
            else pd.DataFrame()
        )
        return df_combined, source_tables

class FullTextCurationTool(AgentTool):
    def __init__(
        self,
        pmid: str,
        cls: Any,
        tool_name: str,
        tool_description: str,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
        pmid_db: PMIDDB | None = None,
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
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.pmid_db = pmid_db if pmid_db is not None else PMIDDB()
        
    def _get_tool_name(self) -> str:
        return self.tool_name

    def _get_tool_description(self) -> str:
        return self.tool_description

    def _run(self, previous_errors: str):
        pmid_info = self.pmid_db.select_pmid_info(self.pmid)
        if pmid_info is None:
            return None, None
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
            previous_errors=previous_errors,
        ), article_text