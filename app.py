import streamlit as st
import logging

from src.paper_analyzer import populate_paper_to_template

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

prompts = """
Please act as a Medical Assistant, extract the following information from the provided biomedical paper and output as a table in markdown format:
1. Population: Describe the patient age distribution, including categories such as "pediatric," "adults," "old adults," "maternal," "fetal," "neonate," etc.
2. Data Source: Identify where the data originated, such as "Statewide surveillance in Victoria, Australia between 1 March 2002 and 31 August 2004." 
3. Inclusion Criteria: Specify the cases that were included in the paper.
4. Exclusion Criteria: Specify the cases that were excluded from the paper.
5. Study Type: Determine the type of study conducted, such as "clinical trial," "pharmacoepidemiology," etc.
6. Outcome: Describe the main outcome of the study, for example, "the disappearance of proteinuria."
7. Study Design: Explain how the study was designed, e.g., "prospective single-arm open-labeled pilot trial."
8. Sample Size: Provide the sample size used in the study.
9. drug names: What drugs mentioned in the paper, like "cefepime", "vancomycin", and so on.
10. pregnancy stage, What pregnancy stages of patients mentioned in the paper, like "postpartum", "before pregnancy", "1st trimester" and so on. If not mentioned, please use 'N/A'.
Please note: only output markdown table without any other characters and embed the text in code chunks, so it won't convert to HTML in the assistant.,
Now, you don't need to response until I post the paper.
"""

ss = st.session_state
ss.setdefault("tables", [])
ss.setdefault("extracted_content", "")
ss.setdefault("prompts", prompts)
ss.setdefault("in_progress", False)

def on_id_changed(pmid: str):
    v = pmid
    the_prompts = ss.get("id-prompts")
    ss.in_progress = True
    (res, choice, code) = populate_paper_to_template(v, the_prompts)
    ss.in_progress = False
    if not res:
        st.error(choice)
        ss.tables = []
        return
    ss.extracted_content = choice.message.content
    # tables = extract_tables_from_html(content)
    # ss.tables = tables
def on_reextract():
    v = on_id_changed()

def escape_markdown(content: str) -> str:
    content = content.replace("#", "\\#")
    content = content.replace("*", "\\*")
    return content

def on_prompts_change():
    v = ss.get("id-prompts")

st.set_page_config(layout="wide")
st.title("Extract Tabula Data")
main_panel, right_panel = st.columns([1, 1])
with main_panel:
    the_pmid = st.text_input(
        label="PMID",
        placeholder="Enter PMID",
        key="id-input",
    )
    submitted = st.button("Extract")
    if the_pmid and submitted:
        with st.spinner("Extracting data ..."):
            on_id_changed(the_pmid)

    if ss.in_progress:
        st.spinner("Extracting data ...")
    tables = ss.get("tables")
    for tbl in tables:
        if "caption" in tbl:
            st.markdown(escape_markdown(tbl["caption"]))
        if "table" in tbl:
            st.dataframe(tbl["table"])
        if "footnote" in tbl:
            st.markdown(escape_markdown(tbl["footnote"]))
        st.divider()
    
    st.divider()
    extracted_content = ss.get("extracted_content")
    st.markdown(extracted_content)
with right_panel:
    st.text_area(
        "prompts", 
        placeholder="Input prompts here", 
        height=500,
        value=ss.prompts,
        on_change=on_prompts_change,
        key="id-prompts"
    )
