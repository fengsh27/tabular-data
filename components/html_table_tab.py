import logging
from typing import Optional
from datetime import datetime
import streamlit as st

from extractor.article_retriever import ExtendArticleRetriever
from extractor.html_table_extractor import HtmlTableExtractor
from extractor.stampers import Stamper
from extractor.utils import escape_markdown

stamper = None
ss = st.session_state
def on_input_changed(pmid: Optional[str]=None):
    if pmid is None:
        pmid = ss.get("id-html-tab-input")
    pmid = pmid.strip()
    stamper.pmid = pmid
    # initialize
    ss.html_info = ""

    # retrieve article
    retriever = ExtendArticleRetriever() # 
    res, html_content, code = retriever.request_article(pmid)
    if not res:
        error_msg = f"Failed to retrieve article. \n {html_content}"
        st.error(error_msg)
        ss.html_retrieved_tables = []
        return
    stamper.output_html(html_content)
    extractor = HtmlTableExtractor()
    retrieved_tables = extractor.extract_tables(html_content)
    ss.html_retrieved_tables = retrieved_tables

    tmp_info = (
        'no table found' 
        if len(retrieved_tables) == 0 
        else f'{len(retrieved_tables)} tables found'
    )
    ss.html_info = f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Retrieving completed, {tmp_info}"

def html_tab(stmpr: Stamper):
    global stamper
    stamper = stmpr
    ss.setdefault("html_retrieved_tables", None)
    ss.setdefault("html_info", "")

    st.title("Extract HTML tables")
    the_pmid = st.text_input(
        label="PMID",
        placeholder="Enter PMID",
        key="id-html-tab-input",
    )
    html_extract_btn = st.button("Extract Tables ...")
    if html_extract_btn:
        with st.spinner("Obtaining article ..."):
            on_input_changed()

    if ss.html_info and len(ss.html_info) > 0:
        st.write(ss.html_info)

    all_tables = ss.html_retrieved_tables
    if all_tables is not None and len(all_tables) > 0:
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
            if "raw_tag" in tbl:
                with st.expander("Html Table"):
                    st.write(tbl["raw_tag"])
            st.divider()

            
