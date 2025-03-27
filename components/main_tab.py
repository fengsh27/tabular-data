import os
from typing import Optional, List, Dict, Callable
import streamlit as st
from datetime import datetime
import logging
from nanoid import generate
from streamlit_modal import Modal
import pandas as pd

import ast
import json
import time

from TabFuncFlow.utils.table_utils import single_html_table_to_markdown
from components.table_utils import select_pk_tables
from extractor.agents.agent_utils import increase_token_usage
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow
from extractor.constants import ( 
    LLM_CHATGPT_4O,
    PROMPTS_NAME_PE,
    PROMPTS_NAME_PK,
    PROMPTS_NAME_PK,
    LLM_GEMINI_PRO,
    LLM_DEEPSEEK_CHAT,
)
from extractor.stampers import ArticleStamper, Stamper
from extractor.article_retriever import ArticleRetriever
from extractor.request_openai import (
    get_openai,
)
from extractor.utils import (
    convert_csv_table_to_dataframe,
    convert_html_to_text,
    escape_markdown,
    extract_table_title,
    is_valid_csv_table,
    preprocess_csv_table_string,
    remove_references,
)
from extractor.html_table_extractor import HtmlTableExtractor
from extractor.prompts_utils import (
    generate_paper_text_prompts,
    generate_tables_prompts,
    generate_question,
    TableExtractionPromptsGenerator,
)
from extractor.generated_table_processor import GeneratedPKSummaryTableProcessor
from extractor.request_geminiai import (
    get_gemini,
)

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
        ss.main_retrieved_tables=[]
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
    step_name: Optional[str]=None, 
    step_description: Optional[str]=None,
    step_output: Optional[str]=None,
    step_reasoning_process: Optional[str]=None,
    token_usage: Optional[dict]=None,
):
    if step_name is not None:
        output_info("=" * 64)
        output_info(step_name)
    if step_description is not None:
        output_info(step_description)
    if token_usage is not None:
        usage_str = f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
        output_info(
            usage_str
        )
        ss.token_usage = increase_token_usage(ss.token_usage, token_usage)
        usage_str = f"overall total tokens: {ss.token_usage['total_tokens']}, overall prompt tokens: {ss.token_usage['prompt_tokens']}, overall completion tokens: {ss.token_usage['completion_tokens']}"
        output_info(
            usage_str
        )
    if step_reasoning_process is not None:
        output_info(f"\n\n{step_reasoning_process}\n\n")
    if step_output is not None:
        output_info(step_output)
    
def on_input_change(pmid: Optional[str]=None):
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
    retriever = ArticleRetriever() # ExtendArticleRetriever() #
    res, html_content, code = retriever.request_article(pmid)
    if not res:
        error_msg = f"Failed to retrieve article. \n {html_content}"
        st.error(error_msg)
        ss.main_retrieved_tables = []
        return
    stamper.output_html(html_content)

    # extract text and tables
    paper_text = convert_html_to_text(html_content)
    paper_text = remove_references(paper_text)
    ss.main_article_text = paper_text
    extractor = HtmlTableExtractor()
    retrieved_tables = extractor.extract_tables(html_content)
    ss.main_retrieved_tables = retrieved_tables
    
    tmp_info = (
        'no table found' 
        if len(retrieved_tables) == 0 
        else f'{len(retrieved_tables)} tables found'
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

    llm = get_openai() if ss.main_llm_option == LLM_CHATGPT_4O else get_gemini()
    ss.token_usage = None
    ss.logs = ""
    if ss.main_prompts_option == PROMPTS_NAME_PK:
        include_tables = ss.main_retrieved_tables
        
        output_info("We are going to select pk summary tables")

        """ Step 1 - Identify PK Tables """
        """ Analyze the given HTML to determine which tables are about PK. """
        """ Example response: ["Table 1", "Table 2"] """
        selected_tables, indexes, token_usage = select_pk_tables(include_tables, llm)
        table_no = []
        for ix in indexes:
            table_no.append(f"Table {int(ix)+1}")

        try:            
            notification = f"From the HTML you selected, the following table(s) are related to PK (Pharmacokinetics): {table_no}"
            
            output_info(notification)
            output_info("Step 1 completed, token usage: " + str(token_usage['total_tokens']))
            st.write(notification)
            st.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Step 1 completed, token usage: {token_usage['total_tokens']}")

        except Exception as e:
            logger.error(e)
            st.error(e)
            return

        """ Step 2 - Further Divide Each Table """
        # step1_content = ['Table II', 'Table III']

        time.sleep(0.1)

        dfs = []
        for table in selected_tables:
            html_table = table["raw_tag"]
            caption = "\n".join([table["caption"], table["footnote"]])
            workflow = PKSumWorkflow(llm=llm)
            workflow.build()
            df = workflow.go(
                html_content=html_table, 
                caption_and_footnote=caption, 
                step_callback=output_step,
            )
            dfs.append(df)
        df_combined = pd.concat(dfs, axis=0)

        output_info(f"Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}")
        st.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting tabular data completed, token usage: {ss.token_usage['total_tokens']}")

        ss.main_extracted_result = df_combined.to_csv()
        ss.main_token_usage = ss.token_usage
        # ss.main_token_usage = sum(step3_usage_list)
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
        'no table found' 
        if len(retrieved_tables) == 0 
        else f'{len(retrieved_tables)} tables found'
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
    ss.setdefault("main_extracted_btn_disabled", True)
    ss.setdefault("main_prompts_option", PROMPTS_NAME_PK)
    ss.setdefault("main_llm_option", LLM_CHATGPT_4O)
    ss.setdefault("logs", "")
    ss.setdefault('token_usage', None)
    
    modal = Modal(
        "Prompts",
        key="prompts-modal",
        padding=10,
        max_width=1200,
    )
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
                'Retrieve Tables ...',
                key='w-pmid-retrieve',
            )
            pmid_extract_btn = st.button(
                'Extract Data...',
                key='w-pmid-extract',
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
                    on_click=on_clear_html_table_input
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
            st.header(f"Extracted Result {'' if usage is None else '(token: '+str(usage)+')'}.",
                      divider="blue")
            if is_valid_csv_table(ss.main_extracted_result):
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
        if (
            ss.main_retrieved_tables is not None and 
            len(ss.main_retrieved_tables) > 0
        ):
            st.subheader("Tables in Article:")
            for ix in range(len(ss.main_retrieved_tables)):
                tbl = ss.main_retrieved_tables[ix]
                st.subheader(f"Table {ix+1}")
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
        llm_option = st.radio("What LLM would you like to use?", (
            LLM_CHATGPT_4O, LLM_GEMINI_PRO, LLM_DEEPSEEK_CHAT,
        ), index=0)
        ss.main_llm_option = llm_option
        st.divider()
        prompts_array = (PROMPTS_NAME_PK, PROMPTS_NAME_PE)
        option = st.selectbox("What type of prompts would you like to use?", prompts_array, index=0)
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
                title = f" - table {ix+1}: {title}" if title is not None else f"table {ix + 1}"
                st.markdown(
                    title,
                    # key=f"w-pmid-tbl-check-{ix}"
                )
    
    if modal.is_open():
        generator = TableExtractionPromptsGenerator(ss.main_prompts_option)
        prmpts = generator.get_prompts_file_content()
        prmpts += "\n\n\n"
        with modal.container():
            st.text(prmpts)
            st.divider()
    
    js = f"""
<script>
    function scroll(dummy_var_to_force_repeat_execution){{
        var textAreas = parent.document.querySelectorAll('.stTextArea textarea'); // document.getElementById("logs_input"); // 
        for (let index = 0; index < textAreas.length; index++) {{
            // textAreas[index].style.color = 'red'
            textAreas[index].scrollTop = textAreas[index].scrollHeight;
        }}
    }}
    scroll({len(ss.logs_input)})
</script>
"""
    st.components.v1.html(js)
