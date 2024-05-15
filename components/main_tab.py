from typing import Optional
import streamlit as st
from datetime import datetime
import logging
from nanoid import generate
from streamlit_modal import Modal

from extractor.constants import ( 
    PROMPTS_NAME_PE,
    PROMPTS_NAME_PK,
    LLM_CHATGPT_35,
    LLM_CHATGPT_40,
    LLM_GEMINI_FLASH,
    LLM_GEMINI_PRO,
)
from extractor.stampers import Stamper
from extractor.article_retriever import ExtendArticleRetriever, ArticleRetriever
from extractor.request_openai import (
    request_to_chatgpt_35,
    request_to_chatgpt_40,
)
from extractor.utils import (
    convert_csv_table_to_dataframe,
    convert_html_to_text,
    escape_markdown,
    extract_table_title,
    is_valid_csv_table,
    remove_references,
)
from extractor.html_table_extractor import HtmlTableExtractor
from extractor.prompts_utils import (
    generate_paper_text_prompts,
    generate_tables_prompts,
    generate_question,
    TableExtractionPromptsGenerator,
)
from extractor.request_geminiai import (
    request_to_gemini_15_pro,
    request_to_gemini_15_flash,
)

logger = logging.getLogger(__name__)

stamper = None
ss = st.session_state

def clear_results(clear_retrieved_table=False):
    ss.main_info = ""
    if clear_retrieved_table:
        ss.main_retrieved_tables=[]
    ss.main_extracted_result = None
    ss.main_token_usage = None

def on_input_change(pmid: Optional[str]=None):
    if pmid is None:
        pmid = ss.get("w-pmid-input")
    pmid = pmid.strip()
    stamper.pmid = pmid
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
    ss.main_info = f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Retrieving completed, {tmp_info}"

def on_extract(pmid: str):
    # initialize
    pmid = pmid.strip()
    stamper.pmid = pmid
    clear_results()

    # prepare prompts including article prmpots and table prompts
    prmpt_generator = TableExtractionPromptsGenerator()
    first_prompots = prmpt_generator.generate_system_prompts(ss.main_prompts_option)

    include_tables = []
    for ix in range(len(ss.main_retrieved_tables)):
        include_tbl = ss.get(f"w-pmid-tbl-check-{ix}")
        if include_tbl:
            include_tables.append(ss.main_retrieved_tables[ix])
    prompts_list = [{"role": "user", "content": first_prompots}]
    if len(include_tables) > 0:
        prompts_list.append({
            "role": "user",
            "content": generate_tables_prompts(include_tables)
        })
    
    customized_prompts = ss.main_customized_prompts
    if customized_prompts and len(customized_prompts) > 0:
        prompts_list.append({
            "role": "user",
            "content": customized_prompts,
        })

    source = ""
    if len(include_tables) > 0:
        source = "tables"
    else:
        st.error("Please select at least one table")
        return
    assert len(prompts_list) > 0
    
    # chat with LLM
    try:
        tmp_prmpts_list = [*prompts_list, {"role": "user", "content": generate_question(source)}]
        stamper.output_prompts(tmp_prmpts_list)
        if ss.main_llm_option == LLM_CHATGPT_40:
            res, content, usage = request_to_chatgpt_40( # request_to_gemini(
                prompts_list,
                generate_question(source),
            )
        elif ss.main_llm_option == LLM_CHATGPT_35:
            res, content, usage = request_to_chatgpt_35(
                prompts_list,
                generate_question(source),
            )
        elif ss.main_llm_option == LLM_GEMINI_FLASH:
            res, content, usage = request_to_gemini_15_flash(
                prompts_list,
                generate_question(source),
            )
        else:
            res, content, usage = request_to_gemini_15_pro(
                prompts_list,
                generate_question(source),
            )
        stamper.output_result(f"{content}\n\nUsage: {str(usage) if usage is not None else ''}")
        ss.main_extracted_result = content
        ss.main_token_usage = usage
        
        ss.main_info = f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting completed"
    except Exception as e:
        logger.error(e)
        st.error(e)
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
    stamper.pmid = pmid
    on_extract(pmid)
    
def main_tab(stmpr: Stamper):
    ss.setdefault("main_info", "")
    ss.setdefault("main_article_text", "")
    ss.setdefault("main_extracted_result", None)
    ss.setdefault("main_token_usage", None)
    ss.setdefault("main_retrieved_tables", None)
    ss.setdefault("main_extracted_btn_disabled", True)
    ss.setdefault("main_prompts_option", PROMPTS_NAME_PK)
    ss.setdefault("main_llm_option", LLM_CHATGPT_40)
    ss.setdefault("main_customized_prompts", "")

    modal = Modal(
        "Prompts",
        key="prompts-modal",
        padding=10,
        max_width=1200,
    )
    global stamper
    stamper = stmpr
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
            usage = ss.main_token_usage
            st.header(f"Extracted Result {'' if usage is None else '(token: '+str(usage)+')'}", divider="blue")
            if is_valid_csv_table(ss.main_extracted_result):
                df = convert_csv_table_to_dataframe(ss.main_extracted_result)
                if df is not None:
                    st.dataframe(df)
                else:
                    st.markdown(ss.main_extracted_result)
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
                    st.dataframe(tbl["table"])
                if "footnote" in tbl:
                    st.markdown(escape_markdown(tbl["footnote"]))
                if "raw_tag" in tbl:
                    with st.expander("Html Table"):
                        st.write(tbl["raw_tag"])
                st.divider()
    with prompts_panel:
        llm_option = st.radio("What LLM would you like to use?", (
            LLM_CHATGPT_40, LLM_CHATGPT_35, LLM_GEMINI_FLASH, LLM_GEMINI_PRO
        ), index=0)
        ss.main_llm_option = llm_option
        st.divider()
        prompts_array = (PROMPTS_NAME_PK, PROMPTS_NAME_PE)
        option = st.selectbox("What type of prompts would you like to use?", prompts_array, index=0)
        ss.main_prompts_option = option
        open_modal = st.button("View Prompts ...")
        st.divider()
        customized_prompts = st.text_area("Customized Prompts (Optional):", height=40)
        ss.main_customized_prompts = customized_prompts
        if not ss.main_extracted_btn_disabled:
            tables = (
                ss.main_retrieved_tables if ss.main_retrieved_tables is not None else []
            )
            for ix in range(len(tables)):
                tbl = tables[ix]
                title = extract_table_title(tbl)
                title = title if title is not None else f"table {ix + 1}"
                st.checkbox(
                    title,
                    key=f"w-pmid-tbl-check-{ix}"
                )
    
    if open_modal:
        modal.open()
    if modal.is_open():
        generator = TableExtractionPromptsGenerator()
        prmpts = generator.get_prompts_file_content(ss.main_prompts_option)
        prmpts += "\n\n\n"
        with modal.container():
            st.text(prmpts)
            st.divider()
