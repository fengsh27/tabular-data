import os
from typing import Optional
import streamlit as st
from datetime import datetime
import logging
from nanoid import generate
import pandas as pd
import time

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.pmid_extractor.table_utils import select_pk_summary_tables, select_pk_demographic_tables
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow
from extractor.agents.pk_individual.pk_ind_workflow import PKIndWorkflow
from extractor.agents.pk_specimen_summary.pk_spec_sum_workflow import PKSpecSumWorkflow
from extractor.agents.pk_population_summary.pk_popu_sum_workflow import PKPopuSumWorkflow
from extractor.agents.pk_drug_summary.pk_drug_sum_workflow import PKDrugSumWorkflow
from extractor.agents.pk_specimen_individual.pk_spec_ind_workflow import PKSpecIndWorkflow
from extractor.agents.pk_population_individual.pk_popu_ind_workflow import PKPopuIndWorkflow
from extractor.agents.pk_drug_individual.pk_drug_ind_workflow import PKDrugIndWorkflow
from extractor.constants import (
    LLM_CHATGPT_4O,
    PROMPTS_NAME_PK_SUM,
    PROMPTS_NAME_PK_IND,
    PROMPTS_NAME_PK_SPEC_SUM,
    PROMPTS_NAME_PK_DRUG_SUM,
    PROMPTS_NAME_PK_POPU_SUM,
    PROMPTS_NAME_PK_SPEC_IND,
    PROMPTS_NAME_PK_DRUG_IND,
    PROMPTS_NAME_PK_POPU_IND,
    LLM_GEMINI_PRO,
    LLM_DEEPSEEK_CHAT,
)
from extractor.stampers import ArticleStamper
from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.request_openai import (
    get_openai,
)
from extractor.request_deepseek import get_deepseek
from extractor.request_geminiai import (
    get_gemini,
)
from extractor.utils import (
    convert_csv_table_to_dataframe,
    convert_html_to_text_no_table,  # Yichuan
    escape_markdown,
    extract_table_title,
    is_valid_csv_table,
    preprocess_csv_table_string,
    remove_references,
)
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor

logger = logging.getLogger(__name__)

output_folder = os.environ.get("TEMP_FOLDER", "./tmp")
stamper_enabled = os.environ.get("LOG_ARTICLE", "FALSE") == "TRUE"
stamper = ArticleStamper(output_folder, stamper_enabled)
ss = st.session_state


def set_stamper_pmid(pmid):
    global stamper
    stamper.pmid = pmid


def clear_results(clear_retrieved_table=False):
    ss.main_info = ""
    if clear_retrieved_table:
        ss.main_retrieved_tables = []
    ss.main_extracted_result = None
    ss.main_token_usage = None
    ss.token_usage = None
    ss.logs = ""


# Define the scroll operation as a function and pass in something unique for each
# page load that it needs to re-evaluate where "bottom" is
def output_info(msg: str):
    ss.logs += "\n" + msg
    ss.logs_input = ss.logs
    logger.info(msg)


def clear_info(msg: str):
    ss.logs = ""
    ss.logs_input = ss.logs


def output_step(
    step_name: Optional[str] = None,
    step_description: Optional[str] = None,
    step_output: Optional[str] = None,
    step_reasoning_process: Optional[str] = None,
    token_usage: Optional[dict] = None,
):
    if step_name is not None:
        output_info("=" * 64)
        output_info(step_name)
    if step_description is not None:
        output_info(step_description)
    if token_usage is not None:
        usage_str = f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
        output_info(usage_str)
        ss.token_usage = increase_token_usage(ss.token_usage, token_usage)
        usage_str = f"overall total tokens: {ss.token_usage['total_tokens']}, overall prompt tokens: {ss.token_usage['prompt_tokens']}, overall completion tokens: {ss.token_usage['completion_tokens']}"
        output_info(usage_str)
    if step_reasoning_process is not None:
        output_info(f"\n\n{step_reasoning_process}\n\n")
    if step_output is not None:
        output_info(step_output)


def on_input_change(pmid: Optional[str] = None):
    output_info("Retrieving tables from article ...")

    global stamper
    if pmid is None:
        pmid = ss.get("w-pmid-input")
    pmid = pmid.strip()
    set_stamper_pmid(pmid)
    # initialize
    clear_results(True)
    ss.main_extracted_btn_disabled = False

    # retrieve article
    retriever = ArticleRetriever()  # ExtendArticleRetriever() #
    res, html_content, code = retriever.request_article(pmid)
    if not res:
        error_msg = f"Failed to retrieve article. \n {html_content}"
        st.error(error_msg)
        ss.main_retrieved_tables = []
        return
    stamper.output_html(html_content)

    # extract text and tables
    # paper_text = convert_html_to_text(html_content)
    paper_text = convert_html_to_text_no_table(html_content)
    paper_text = remove_references(paper_text)
    ss.main_article_text = paper_text
    extractor = HtmlTableExtractor()
    retrieved_tables = extractor.extract_tables(html_content)
    ss.main_retrieved_tables = retrieved_tables
    ss.main_retrieved_title = extractor.extract_title(html_content)
    ss.main_retrieved_abstract = extractor.extract_abstract(html_content)
    ss.main_retrieved_sections = extractor.extract_sections(html_content)

    tmp_info = (
        "no table found"
        if len(retrieved_tables) == 0
        else f"{len(retrieved_tables)} tables found"
    )
    result_str = f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Retrieving completed, {tmp_info}"
    ss.main_info = result_str
    output_info(result_str)


def on_extract(pmid: str):
    global stamper

    # initialize
    pmid = pmid.strip()
    set_stamper_pmid(pmid)
    clear_results()

    llm = (
        get_openai()
        if ss.main_llm_option == LLM_CHATGPT_4O
        else get_gemini()
        if ss.main_llm_option == LLM_GEMINI_PRO
        else get_deepseek()
    )
    ss.token_usage = None
    ss.logs = ""
    if ss.main_prompts_option == PROMPTS_NAME_PK_SUM:
        include_tables = ss.main_retrieved_tables

        output_info("We are going to select pk summary tables")

        """ Step 1 - Identify PK Tables """
        """ Analyze the given HTML to determine which tables are about PK. """
        """ Example response: ["Table 1", "Table 2"] """
        selected_tables, indexes, token_usage = select_pk_summary_tables(
            include_tables, llm
        )
        table_no = []
        for ix in indexes:
            table_no.append(f"Table {int(ix)+1}")

        try:
            if len(table_no) == 0:
                notification = "After analyzing the provided content, none of the tables contain pharmacokinetic (PK) data or ADME properties."
            else:
                notification = f"From the paper you selected, the following table(s) are related to PK (Pharmacokinetics): {table_no}"

            output_info(notification)
            output_info(
                "Step 1 completed, token usage: " + str(token_usage["total_tokens"])
            )
            st.write(notification)
            st.write(
                f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Step 1 completed, token usage: {token_usage['total_tokens']}"
            )

        except Exception as e:
            logger.error(e)
            st.error(e)
            return

        """ Step 2 - Workflow """
        time.sleep(0.1)

        dfs = []
        for table in selected_tables:
            df_table = table["table"]
            caption = "\n".join([table["caption"], table["footnote"]])
            workflow = PKSumWorkflow(llm=llm)
            workflow.build()
            df = workflow.go_md_table(
                title=ss.main_retrieved_title,
                md_table=dataframe_to_markdown(df_table),
                caption_and_footnote=caption,
                step_callback=output_step,
            )
            dfs.append(df)
        df_combined = (
            pd.concat(dfs, axis=0).reset_index(drop=True)
            if len(dfs) > 0
            else pd.DataFrame()
        )

        ss.token_usage = (
            ss.token_usage if ss.token_usage is not None else {**DEFAULT_TOKEN_USAGE}
        )
        output_info(
            f"Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )
        st.write(
            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )

        ss.main_extracted_result = df_combined
        ss.main_token_usage = ss.token_usage
        # ss.main_token_usage = sum(step3_usage_list)
        return
    elif ss.main_prompts_option == PROMPTS_NAME_PK_IND:
        include_tables = ss.main_retrieved_tables

        output_info("We are going to select pk individual tables")

        """ Step 1 - Identify PK Tables """
        """ REUSE PK SUM """
        """ Analyze the given HTML to determine which tables are about PK. """
        """ Example response: ["Table 1", "Table 2"] """
        selected_tables, indexes, token_usage = select_pk_summary_tables(
            include_tables, llm
        )
        table_no = []
        for ix in indexes:
            table_no.append(f"Table {int(ix)+1}")

        try:
            if len(table_no) == 0:
                notification = "After analyzing the provided content, none of the tables contain pharmacokinetic (PK) data or ADME properties."
            else:
                notification = f"From the paper you selected, the following table(s) are related to PK (Pharmacokinetics): {table_no}"

            output_info(notification)
            output_info(
                "Step 1 completed, token usage: " + str(token_usage["total_tokens"])
            )
            st.write(notification)
            st.write(
                f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Step 1 completed, token usage: {token_usage['total_tokens']}"
            )

        except Exception as e:
            logger.error(e)
            st.error(e)
            return

        """ Step 2 - Workflow """
        time.sleep(0.1)

        dfs = []
        for table in selected_tables:
            df_table = table["table"]
            caption = "\n".join([table["caption"], table["footnote"]])
            workflow = PKIndWorkflow(llm=llm)
            workflow.build()
            df = workflow.go_md_table(
                title=ss.main_retrieved_title,
                md_table=dataframe_to_markdown(df_table),
                caption_and_footnote=caption,
                step_callback=output_step,
            )
            dfs.append(df)
        # return
        df_combined = (
            pd.concat(dfs, axis=0).reset_index(drop=True)
            if len(dfs) > 0
            else pd.DataFrame()
        )

        ss.token_usage = (
            ss.token_usage if ss.token_usage is not None else {**DEFAULT_TOKEN_USAGE}
        )
        output_info(
            f"Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )
        st.write(
            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )

        ss.main_extracted_result = df_combined
        ss.main_token_usage = ss.token_usage
        return
    elif ss.main_prompts_option == PROMPTS_NAME_PK_SPEC_SUM:
        output_info("We are going to clean the original text")
        """ Step 1 - Clean Text (through extractor) """
        """ LLM-based text clean has been deprecated due to significant omissions in its output. """
        """ beautifulsoup-based text clean """
        sections = ss.main_retrieved_sections
        if len(sections) == 0:
            notification = "No valid sections were extracted from the text."
        else:
            section_names = [sec["section"] for sec in sections]
            notification = f"Extracted the following sections: {section_names}"
        output_info(notification)
        output_info("Text cleaning completed, token usage: 0")
        # st.write(notification)
        st.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Text cleaning completed, token usage: 0")
        article_content = "\n".join(sec["section"] + "\n" + sec["content"] + "\n" for sec in sections)

        """ Step 2 - Workflow """
        time.sleep(0.1)

        workflow = PKSpecSumWorkflow(llm=llm)
        workflow.build()
        df_combined = workflow.go_full_text(
            title=ss.main_retrieved_title,
            full_text=article_content,
            step_callback=output_step,
        )

        ss.token_usage = (
            ss.token_usage if ss.token_usage is not None else {**DEFAULT_TOKEN_USAGE}
        )
        output_info(
            f"Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )
        st.write(
            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )

        ss.main_extracted_result = df_combined
        ss.main_token_usage = ss.token_usage
        return
    elif ss.main_prompts_option == PROMPTS_NAME_PK_DRUG_SUM:
        output_info("We are going to clean the original text")
        """ Step 1 - Clean Text (through extractor) """
        """ LLM-based text clean has been deprecated due to significant omissions in its output. """
        """ beautifulsoup-based text clean """
        sections = ss.main_retrieved_sections
        if len(sections) == 0:
            notification = "No valid sections were extracted from the text."
        else:
            section_names = [sec["section"] for sec in sections]
            notification = f"Extracted the following sections: {section_names}"
        output_info(notification)
        output_info("Text cleaning completed, token usage: 0")
        # st.write(notification)
        st.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Text cleaning completed, token usage: 0")
        article_content = "\n".join(sec["section"] + "\n" + sec["content"] + "\n" for sec in sections)

        """ Step 2 - Workflow """
        time.sleep(0.1)

        workflow = PKDrugSumWorkflow(llm=llm)
        workflow.build()
        df_combined = workflow.go_full_text(
            title=ss.main_retrieved_title,
            full_text=article_content,
            step_callback=output_step,
        )

        ss.token_usage = (
            ss.token_usage if ss.token_usage is not None else {**DEFAULT_TOKEN_USAGE}
        )
        output_info(
            f"Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )
        st.write(
            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )

        ss.main_extracted_result = df_combined
        ss.main_token_usage = ss.token_usage
        return
    elif ss.main_prompts_option == PROMPTS_NAME_PK_POPU_SUM:
        output_info("We are going to clean the original text")
        """ Step 1 - Clean Text (through extractor) """
        """ LLM-based text clean has been deprecated due to significant omissions in its output. """
        """ beautifulsoup-based text clean """
        sections = ss.main_retrieved_sections
        if sections is None:
            notification = "Please retrieve the article first."
        if len(sections) == 0:
            notification = "No valid sections were retrieved from the text."
        else:
            section_names = [sec["section"] for sec in sections]
            notification = f"The following sections were successfully retrieved: {section_names}"
        output_info(notification)
        output_info("Text cleaning completed, token usage: 0")
        # st.write(notification)
        st.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Text cleaning completed, token usage: 0")
        article_content = "\n".join(sec["section"] + "\n" + sec["content"] + "\n" for sec in sections)

        """ Step 2 - Workflow """
        time.sleep(0.1)

        workflow = PKPopuSumWorkflow(llm=llm)
        workflow.build()
        df_combined = workflow.go_full_text(
            title=ss.main_retrieved_title,
            full_text=article_content,
            step_callback=output_step,
        )

        ss.token_usage = (
            ss.token_usage if ss.token_usage is not None else {**DEFAULT_TOKEN_USAGE}
        )
        output_info(
            f"Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )
        st.write(
            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )

        ss.main_extracted_result = df_combined
        ss.main_token_usage = ss.token_usage
        return
    elif ss.main_prompts_option == PROMPTS_NAME_PK_SPEC_IND:
        output_info("We are going to clean the original text")
        """ Step 1 - Clean Text (through extractor) """
        """ LLM-based text clean has been deprecated due to significant omissions in its output. """
        """ beautifulsoup-based text clean """
        sections = ss.main_retrieved_sections
        if len(sections) == 0:
            notification = "No valid sections were extracted from the text."
        else:
            section_names = [sec["section"] for sec in sections]
            notification = f"Extracted the following sections: {section_names}"
        output_info(notification)
        output_info("Text cleaning completed, token usage: 0")
        # st.write(notification)
        st.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Text cleaning completed, token usage: 0")
        article_content = "\n".join(sec["section"] + "\n" + sec["content"] + "\n" for sec in sections)

        """ Step 2 - Workflow """
        time.sleep(0.1)

        workflow = PKSpecIndWorkflow(llm=llm)
        workflow.build()
        df_combined = workflow.go_full_text(
            title=ss.main_retrieved_title,
            full_text=article_content,
            step_callback=output_step,
        )

        ss.token_usage = (
            ss.token_usage if ss.token_usage is not None else {**DEFAULT_TOKEN_USAGE}
        )
        output_info(
            f"Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )
        st.write(
            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )

        ss.main_extracted_result = df_combined
        ss.main_token_usage = ss.token_usage
        return
    elif ss.main_prompts_option == PROMPTS_NAME_PK_DRUG_IND:
        output_info("We are going to clean the original text")
        """ Step 1 - Clean Text (through extractor) """
        """ LLM-based text clean has been deprecated due to significant omissions in its output. """
        """ beautifulsoup-based text clean """
        sections = ss.main_retrieved_sections
        if len(sections) == 0:
            notification = "No valid sections were extracted from the text."
        else:
            section_names = [sec["section"] for sec in sections]
            notification = f"Extracted the following sections: {section_names}"
        output_info(notification)
        output_info("Text cleaning completed, token usage: 0")
        # st.write(notification)
        st.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Text cleaning completed, token usage: 0")
        article_content = "\n".join(sec["section"] + "\n" + sec["content"] + "\n" for sec in sections)

        """ Step 2 - Workflow """
        time.sleep(0.1)

        workflow = PKDrugIndWorkflow(llm=llm)
        workflow.build()
        df_combined = workflow.go_full_text(
            title=ss.main_retrieved_title,
            full_text=article_content,
            step_callback=output_step,
        )

        ss.token_usage = (
            ss.token_usage if ss.token_usage is not None else {**DEFAULT_TOKEN_USAGE}
        )
        output_info(
            f"Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )
        st.write(
            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )

        ss.main_extracted_result = df_combined
        ss.main_token_usage = ss.token_usage
        return
    elif ss.main_prompts_option == PROMPTS_NAME_PK_POPU_IND:
        # include_tables = ss.main_retrieved_tables
        # output_info("We are going to select pk population individual tables")
        # """ Step 1 - Identify population Tables """
        # """ Analyze the given HTML to determine which tables are about PK demographic data. """
        # """ Example response: ["Table 1", "Table 2"] """
        # selected_tables, indexes, token_usage = select_pk_demographic_tables(
        #     include_tables, llm
        # )
        # table_no = []
        # for ix in indexes:
        #     table_no.append(f"Table {int(ix)+1}")
        #
        # try:
        #     if len(table_no) == 0:
        #         notification = "After analyzing the provided content, none of the tables contain pharmacokinetic demographic data."
        #     else:
        #         notification = f"From the paper you selected, the following table(s) are related to pharmacokinetic demographic data: {table_no}"
        #
        #     output_info(notification)
        #     output_info(
        #         "Step 1 completed, token usage: " + str(token_usage["total_tokens"])
        #     )
        #     st.write(notification)
        #     st.write(
        #         f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Step 1 completed, token usage: {token_usage['total_tokens']}"
        #     )
        #
        # except Exception as e:
        #     logger.error(e)
        #     st.error(e)
        #     return
        output_info("We are going to clean the original text")
        """ Step 1 - Clean Text (through extractor) """
        """ LLM-based text clean has been deprecated due to significant omissions in its output. """
        """ beautifulsoup-based text clean """
        sections = ss.main_retrieved_sections
        if sections is None:
            notification = "Please retrieve the article first."
        if len(sections) == 0:
            notification = "No valid sections were retrieved from the text."
        else:
            section_names = [sec["section"] for sec in sections]
            notification = f"The following sections were successfully retrieved: {section_names}"
        output_info(notification)
        output_info("Text cleaning completed, token usage: 0")
        # st.write(notification)
        st.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Text cleaning completed, token usage: 0")
        article_content = "\n".join(sec["section"] + "\n" + sec["content"] + "\n" for sec in sections)

        def remove_duplicate_lines(article_content):
            """
            Remove duplicate lines from article content while preserving:
            - Table rows (lines containing '|')
            - Empty lines (they are ignored, not treated as duplicates)
            """
            seen = set()
            result = []
            for line in article_content.splitlines():
                if '|' in line:
                    result.append(line)
                elif line not in seen:
                    seen.add(line)
                    result.append(line)
            return '\n'.join(result)
        _article_content = remove_duplicate_lines(article_content)
        if _article_content != article_content:
            article_content = _article_content
            output_info("Duplicate content has been filtered out.")
            st.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Duplicate content has been filtered out.")

        """ Step 2 - Workflow """
        time.sleep(0.1)

        workflow = PKPopuIndWorkflow(llm=llm)
        workflow.build()
        df_combined = workflow.go_full_text(
            title=ss.main_retrieved_title,
            full_text=article_content,
            step_callback=output_step,
        )

        ss.token_usage = (
            ss.token_usage if ss.token_usage is not None else {**DEFAULT_TOKEN_USAGE}
        )
        output_info(
            f"Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )
        st.write(
            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}"
        )

        ss.main_extracted_result = df_combined
        ss.main_token_usage = ss.token_usage
        return


def on_retrive_table_from_html_table(html_table: str):
    html_table = html_table.strip()
    if len(html_table) == 0:
        return
    clear_results(True)
    extractor = HtmlTableExtractor()
    retrieved_tables = extractor.extract_tables(html_table)
    ss.main_retrieved_tables = retrieved_tables
    ss.main_extracted_btn_disabled = False
    tmp_info = (
        "no table found"
        if len(retrieved_tables) == 0
        else f"{len(retrieved_tables)} tables found"
    )
    ss.main_info = f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Retrieving completed, {tmp_info}"


def on_extract_from_html_table():
    pmid = generate(size=10)
    set_stamper_pmid(pmid)
    on_extract(pmid)


def main_tab():
    ss.setdefault("main_info", "")
    ss.setdefault("main_article_text", "")
    ss.setdefault("main_extracted_result", None)
    ss.setdefault("main_token_usage", None)
    ss.setdefault("main_retrieved_tables", None)
    ss.setdefault("main_retrieved_title", None)
    ss.setdefault("main_retrieved_abstract", None)  # Yichuan 0501
    ss.setdefault("main_retrieved_sections", None)  # Yichuan 0502
    ss.setdefault("main_extracted_btn_disabled", True)
    ss.setdefault("main_prompts_option", PROMPTS_NAME_PK_SUM)
    ss.setdefault("main_llm_option", LLM_CHATGPT_4O)
    ss.setdefault("logs", "")
    ss.setdefault("token_usage", None)

    # Note: The modal functionality below is currently unused. - Yichuan
    # modal = Modal(
    #     "Prompts",
    #     key="prompts-modal",
    #     padding=10,
    #     max_width=1200,
    # )
    global stamper
    st.title("Extract Tabular Data")
    extracted_panel, prompts_panel = st.columns([2, 1])
    with extracted_panel:
        with st.expander("Input PMID/PMCID"):
            the_pmid = st.text_input(
                # the_pmid = st.text_input(
                label="PMID/PMCID",
                placeholder="Enter PMID or PMCID",
                key="w-pmid-input",
            )
            pmid_retrieve_btn = st.button(
                # retrieve_btn = st.button(
                "Retrieve Article...",
                key="w-pmid-retrieve",
            )
            pmid_extract_btn = st.button(
                "Extract Data...",
                key="w-pmid-extract",
            )

            if the_pmid and pmid_retrieve_btn:
                with st.spinner("Obtaining article ..."):
                    on_input_change(the_pmid)
            if the_pmid and pmid_extract_btn:
                with st.spinner("Extracting data ..."):
                    on_extract(the_pmid)
        with st.expander("Input Html table or html article"):
            text_area, clear_area = st.columns([5, 1])
            with text_area:
                html_table_input = st.text_area(
                    label="html table or html article",
                    height=200,
                    key="html_table_input",
                )
            with clear_area:

                def on_clear_html_table_input():
                    ss.html_table_input = ""

                clear_btn = st.button(
                    "clear",
                    help="Clear html table or html article",
                    on_click=on_clear_html_table_input,
                )
            html_table_retrive_btn = st.button(
                "Retrieve Tables ...",
                key="w-html-table-retrieve",
            )
            html_table_extract_btn = st.button(
                "Extract Data ...",
                key="w-html-table-extract",
            )
            if html_table_input and html_table_retrive_btn:
                with st.spinner("Obtaining article ..."):
                    on_retrive_table_from_html_table(html_table_input)
            if html_table_input and html_table_extract_btn:
                with st.spinner("Extract data ..."):
                    on_extract_from_html_table()

        if ss.main_info and len(ss.main_info) > 0:
            st.write(ss.main_info)
        if ss.main_extracted_result is not None:
            usage = ss.main_token_usage["total_tokens"]
            st.header(
                f"Extracted Result {'' if usage is None else '(token: '+str(usage)+')'}.",
                divider="blue",
            )
            if isinstance(ss.main_extracted_result, pd.DataFrame):
                st.dataframe(ss.main_extracted_result)
            elif is_valid_csv_table(ss.main_extracted_result):
                preprocess_csv_table_string(ss.main_extracted_result)
                try:
                    df = convert_csv_table_to_dataframe(ss.main_extracted_result)
                    if df is not None:
                        st.dataframe(df)
                    else:
                        st.markdown(ss.main_extracted_result)
                except Exception as e:
                    st.markdown(str(e))
            else:
                st.markdown(ss.main_extracted_result)
            # st.markdown(ss.main_extracted_result)
            st.divider()
        if ss.main_retrieved_title is not None and len(ss.main_retrieved_title) > 0:
            st.subheader(escape_markdown(ss.main_retrieved_title))
        if ss.main_retrieved_abstract is not None and len(ss.main_retrieved_abstract) > 0:
            st.subheader("Abstract:")
            st.markdown(escape_markdown(ss.main_retrieved_abstract))
        if ss.main_retrieved_tables is not None and len(ss.main_retrieved_tables) > 0:
            st.subheader("Tables:")
            for ix in range(len(ss.main_retrieved_tables)):
                tbl = ss.main_retrieved_tables[ix]
                st.markdown(f"##### Table {ix+1}")
                if "caption" in tbl:
                    st.markdown(escape_markdown(tbl["caption"]))
                if "table" in tbl:
                    try:
                        st.dataframe(tbl["table"])
                    except Exception as e:
                        logger.error(str(e))
                        print("[fengsh] dataframe(table) error")
                if "footnote" in tbl:
                    st.markdown(escape_markdown(tbl["footnote"]))
                if "raw_tag" in tbl:
                    with st.expander("Html Table"):
                        st.write(tbl["raw_tag"])
                st.divider()
    with prompts_panel:
        llm_option = st.radio(
            "What LLM would you like to use?",
            (
                LLM_CHATGPT_4O,
                LLM_DEEPSEEK_CHAT,
            ),
            index=0,
        )
        ss.main_llm_option = llm_option
        st.divider()
        prompts_array = (PROMPTS_NAME_PK_SUM, PROMPTS_NAME_PK_SPEC_SUM, PROMPTS_NAME_PK_DRUG_SUM, PROMPTS_NAME_PK_POPU_SUM,
                         PROMPTS_NAME_PK_IND, PROMPTS_NAME_PK_SPEC_IND, PROMPTS_NAME_PK_DRUG_IND, PROMPTS_NAME_PK_POPU_IND)  #, PROMPTS_NAME_PE)
        option = st.selectbox(
            "What type of prompts would you like to use?", prompts_array, index=0
        )
        ss.main_prompts_option = option
        logs_input = st.text_area("Logs", key="logs_input", height=300)
        st.divider()
        if not ss.main_extracted_btn_disabled:
            tables = (
                ss.main_retrieved_tables if ss.main_retrieved_tables is not None else []
            )
            for ix in range(len(tables)):
                tbl = tables[ix]
                title = extract_table_title(tbl)
                title = (
                    f" - table {ix+1}: {title}"
                    if title is not None
                    else f"table {ix + 1}"
                )
                st.markdown(
                    title,
                    # key=f"w-pmid-tbl-check-{ix}"
                )

    # Note: The modal functionality below is currently unused. - Yichuan
    # if modal.is_open():
    #     generator = TableExtractionPromptsGenerator(ss.main_prompts_option)
    #     prmpts = generator.get_prompts_file_content()
    #     prmpts += "\n\n\n"
    #     with modal.container():
    #         st.text(prmpts)
    #         st.divider()

    js = f"""
<script>
    function scroll(dummy_var_to_force_repeat_execution){{
        var textAreas = parent.document.querySelectorAll('.stTextArea textarea'); // document.getElementById("logs_input"); // 
        for (let index = 0; index < textAreas.length; index++) {{
            textAreas[index].scrollTop = textAreas[index].scrollHeight;
        }}
    }}
    scroll({len(ss.logs_input)})
</script>
"""
    st.components.v1.html(js)

    # Inject custom CSS to change text_input font size
    st.markdown(
        """
<style>
    /* Target the input field inside stTextInput */
    .stTextInput input {
        font-size: 24px !important;  /* Adjust size as needed */
    }
    .stButton button {
        font-size: 24px !important;
    }
</style>
""",
        unsafe_allow_html=True,
    )
