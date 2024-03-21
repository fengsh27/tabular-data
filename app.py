from typing import Optional
import streamlit as st
import streamlit_antd_components as sac
import logging
from dotenv import load_dotenv
from datetime import datetime

from src.request_geminiai import request_to_gemini
from src.utils import convert_html_to_text, remove_references
load_dotenv()

from src.article_retriever import ArticleRetriever, ExtendArticleRetriever
from src.request_openai import request_to_chatgpt
from src.article_stamper import ArticleStamper
from src.prompts_utils import (
    generate_paper_text_prompts,
    generate_tables_prompts,
    generate_question,
    DEFAULT_PROMPTS
)
from src.html_table_extractor import HtmlTableExtractor

def initialize():
    # prepare logger
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler("./logs/app.log")
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # datefmt="%Y-%m-%d %H:%M:%S,uuu"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

initialize()
logger = logging.getLogger(__name__)

ss = st.session_state
ss.setdefault("retrieved_tables", [])
ss.setdefault("prompts", DEFAULT_PROMPTS)
ss.setdefault("disable_extractbtn", True)
ss.setdefault("paper_text", "")
ss.setdefault("extracted_result", None)
ss.setdefault("extracted_result_token_usage", None)
ss.setdefault("tables_tab_tables", [])
ss.setdefault("tbl_info", "")
ss.setdefault("tbl_article_image", None)

def on_id_tabula_tab_input_changed(pmid: Optional[str]=None):
    if pmid is None:
        pmid = ss.get("id-tabula-tab-input")
    ss.extracted_result = None
    ss.extracted_result_token_usage = None
    pmid = pmid.strip()
    ss.disable_extractbtn = True
    ss.tbl_info = ""
    stamper = ArticleStamper(pmid)
    retriever = ExtendArticleRetriever() # ArticleRetriever() #
    res, html_content, code = retriever.request_article(pmid)
    if not res:
        error_msg = f"Failed to retrieve article. \n {html_content}"
        st.error(error_msg)
        ss.retrieved_tables = []
        return
    stamper.output_html(html_content)
    paper_text = convert_html_to_text(html_content)
    paper_text = remove_references(paper_text)
    ss.paper_text = paper_text
    extractor = HtmlTableExtractor()
    retrieved_tables = extractor.extract_tables(html_content)
    ss.retrieved_tables = retrieved_tables
    ss.disable_extractbtn = False
    tmp_info = 'no table found' if len(retrieved_tables) == 0 else f'{len(retrieved_tables)} tables found'
    ss.tbl_info = f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Retrieving completed, {tmp_info}"

def on_extract(pmid: str):
    pmid = pmid.strip()
    ss.extracted_result = None
    ss.extracted_result_token_usage = None
    stamper = ArticleStamper(pmid)
    first_prompots = ss.get("id-prompts")
    first_prompots = \
        first_prompots if len(first_prompots) > 0 else default_prompts
    include_article = ss.get("article-text")
    include_tables = []
    for ix in range(len(ss.retrieved_tables)):
        include_tbl = ss.get(f"tbl-check-{ix}")
        if include_tbl:
            include_tables.append(ss.retrieved_tables[ix])
    prompts_list = [{"role": "user", "content": first_prompots}]
    if include_article:
        text = ss.paper_text
        prompts_list.append({
            "role": "user", 
            "content": generate_paper_text_prompts(text),
        })
    if len(include_tables) > 0:
        prompts_list.append({
            "role": "user",
            "content": generate_tables_prompts(include_tables)
        })
        
    source = ""
    if include_article and len(include_tables) > 0:
        source = "article and tables"
    elif include_article:
        source = "article"
    elif len(include_tables) > 0:
        source = "tables"
    else:
        st.error("Please select article or at least a table")
        return
    assert len(prompts_list) > 0
    try:
        stamper.output_prompts(prompts_list)
        res, content, usage = request_to_gemini(
            prompts_list,
            generate_question(source),
        )
        stamper.output_result(f"{content}\n\nUsage: {str(usage) if usage is not None else ''}")
        ss.extracted_result = content
        ss.extracted_result_token_usage = usage
    except Exception as e:
        logger.error(e)
        st.error(e)
        return

def escape_markdown(content: str) -> str:
    content = content.replace("#", "\\#")
    content = content.replace("*", "\\*")
    return content

def on_prompts_change():
    v = ss.get("id-prompts")

## Components

st.set_page_config(layout="wide")

tab1, tab2 = st.tabs(["Extract Tabula Data", "Extract html table"])

with tab1:
    st.title("Extract Tabula Data")
    main_panel, right_panel = st.columns([2, 1])
    with main_panel:
        the_pmid = st.text_input(
            label="PMID",
            placeholder="Enter PMID",
            key="id-tabula-tab-input",            
        )
        retrieve_btn = st.button(
            'Retrieve Article ...',
            key='retrieve_article',
        )
        extract_btn = st.button(
            'Extract ...',
            key='extract_btn',
        )
            
        if the_pmid and retrieve_btn:
            with st.spinner("Obtaining article ..."):
                on_id_tabula_tab_input_changed(the_pmid)
        if the_pmid and extract_btn:
            with st.spinner("Extracting data ..."):
                on_extract(the_pmid)
        
        info = ss.get("tbl_info")
        st.write(info)
    
        if ss.extracted_result is not None:
            usage = ss.extracted_result_token_usage
            st.header(f"Extracted Result {'' if usage is None else '(token: '+str(usage)+')'}", divider="blue")
            st.markdown(ss.extracted_result)
            st.divider()
        retrieved_tables = ss.get("retrieved_tables")
        if len(retrieved_tables) > 0:
            st.subheader("Tables in article:")
        for tbl in retrieved_tables:
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
    with right_panel:
        st.text_area(
            "prompts", 
            placeholder="Input prompts here", 
            height=500,
            value=ss.prompts,
            on_change=on_prompts_change,
            key="id-prompts"
        )
        retrieved_tables = ss.get("retrieved_tables")
        if not ss.disable_extractbtn:
            st.checkbox(
                "article",
                value=True,
                key="article-text",
            )
        for ix in range(len(retrieved_tables)):
            st.checkbox(
                f"table {ix+1}",  
                key=f"tbl-check-{ix}"
            )

with tab2:
    # callbacks
    def on_id_html_tab_input_changed(pmid: Optional[str]=None):
        if pmid is None:
            pmid = ss.get("id-html-tab-input")
        retriever = ArticleRetriever() # FakeArticleRetriver() # 
        res, html_content, code = retriever.request_article(pmid)
        extractor = HtmlTableExtractor()
        retrieved_tables = extractor.extract_tables(html_content)
        ss.tables_tab_tables = retrieved_tables

    st.title("Extract html table")
    the_pmid = st.text_input(
        label="PMID",
        placeholder="Enter PMID",
        key="id-html-tab-input",
        on_change=on_id_html_tab_input_changed
    )
    html_extract_btn = st.button("Extract Tables ...")
    if html_extract_btn:
        on_id_html_tab_input_changed()

    all_tables = ss.get("tables_tab_tables")
    if len(all_tables) > 0:
        st.subheader("Tables in article:")
    ix = 1
    for tbl in all_tables:
        st.text(f"table {ix}")
        ix += 1
        if "caption" in tbl:
            st.markdown(escape_markdown(tbl["caption"]))
        if "table" in tbl:
            st.dataframe(tbl["table"])
        if "footnote" in tbl:
            st.markdown(escape_markdown(tbl["footnote"]))
        st.divider()