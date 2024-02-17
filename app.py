import streamlit as st
import logging
from src.request_paper import (
    request_paper, 
    extract_tables_from_html,
    convert_table_to_dataframe,
)

logger = logging.getLogger(__name__)

ss = st.session_state
ss.setdefault("tables", [])

def on_id_changed():
    v = ss.get("id-input")
    (res, content, code) = request_paper(v)
    if not res:
        st.error(content)
        ss.tables = []
        return
    tables = extract_tables_from_html(content)
    ss.tables = tables


st.title("Extract Tabula Data")
st.text_input(
    label="PMID",
    placeholder="Enter PMID",
    on_change=on_id_changed,
    key="id-input"
)
tables = ss.get("tables")
for tbl in tables:
    if "caption" in tbl:
        st.markdown(tbl["caption"])
    if "table" in tbl:
        st.dataframe(tbl["table"])
    if "footnote" in tbl:
        st.markdown(tbl["footnote"])
    st.divider()