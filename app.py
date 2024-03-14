from typing import Optional
import streamlit as st
import streamlit_antd_components as sac
import logging
from dotenv import load_dotenv
load_dotenv()

from src.paper_analyzer import populate_paper_to_template
from src.request_paper import PaperRetriver, FakePaperRetriver
from src.request_openai import request_to_chatgpt
from src.article_stamper import AricleStamper
from src.prompts_utils import (
    generate_paper_text_prompts,
    generate_tables_prompts,
    generate_question,
)

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

default_prompts = """
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


Please note: 

1. Only output markdown table without any other characters and embed the text in code chunks, so it won't convert to HTML in the assistant.
2. Ensure to extract all available information for each field without omitting any details.
3. If the information not provided, please leave it empty 
Now, you don't need to response until I post the paper.
"""

ss = st.session_state
ss.setdefault("tables", [])
ss.setdefault("prompts", default_prompts)
ss.setdefault("disable_extractbtn", True)
ss.setdefault("paper_text", "")
ss.setdefault("extracted_result", None)

def on_id_tabula_tab_input_changed(pmid: Optional[str]=None):
    v = pmid if pmid is not None else ss.get("id-tabula-tab-input")
    ss.disable_extractbtn = True
    # (res, choice, code) = populate_paper_to_template(v, the_prompts)
    stamper = AricleStamper(v)
    retriever = FakePaperRetriver(stamper)
    res, html_content, code = retriever.request_paper(v)
    if not res:
        st.error(html_content)
        ss.tables = []
        return
    # stamper.output_html(html_content)
    ss.paper_text = retriever.convert_html_to_text(html_content)
    ss.paper_text = retriever.remove_references(ss.paper_text)
    ss.tables = retriever.extract_tables_from_html(html_content)
    ss.disable_extractbtn = False
    
    #ss.extracted_content = choice.message.content
    # tables = extract_tables_from_html(content)
    # ss.tables = tables

def on_extract(pmid: str):
    ss.extracted_result = None
    stamper = AricleStamper(pmid)
    first_prompots = ss.get("id-prompts")
    first_prompots = \
        first_prompots if len(first_prompots) > 0 else default_prompts
    include_article = ss.get("article-text")
    include_tables = []
    for ix in range(len(ss.tables)):
        include_tbl = ss.get(f"tbl-check-{ix}")
        if include_tbl:
            include_tables.append(ss.tables[ix])
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
        res, choice, code = request_to_chatgpt(
            prompts_list,
            generate_question(source),
            stamper
        )
        ss.extracted_result = choice.message.content
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
            on_change=on_id_tabula_tab_input_changed
        )
        btn = sac.buttons(
            items=[
                sac.ButtonsItem(label='Retrieve Article ...'),
                sac.ButtonsItem(label='Extract ...', disabled=ss.disable_extractbtn)
            ],
            index=0,
            format_func='title',
            align='start',
            direction='horizontal',
            radius='lg',
            return_index=False,
        )
    
        if the_pmid and btn == 'Retrieve Article ...':
            with st.spinner("Obtaining article ..."):
                on_id_tabula_tab_input_changed(the_pmid)
        if the_pmid and btn == 'Extract ...':
            with st.spinner("Extracting data ..."):
                on_extract(the_pmid)
    
        # if ss.in_progress:
        #     st.spinner("Extracting data ...")
    
        if ss.extracted_result is not None:
            st.header("Extracted Result:", divider="blue")
            st.markdown(ss.extracted_result)
            st.divider()
        tables = ss.get("tables")
        if len(tables) > 0:
            st.subheader("Tables in article:")
        for tbl in tables:
            if "caption" in tbl:
                st.markdown(escape_markdown(tbl["caption"]))
            if "table" in tbl:
                st.dataframe(tbl["table"])
            if "footnote" in tbl:
                st.markdown(escape_markdown(tbl["footnote"]))
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
        tables = ss.get("tables")
        if not ss.disable_extractbtn:
            st.checkbox(
                "article",
                value=True,
                key="article-text",
            )
        for ix in range(len(tables)):
            st.checkbox(
                f"table {ix+1}",  
                key=f"tbl-check-{ix}",
                on_change=on_table_selected
            )

with tab2:
    # callbacks
    def on_id_html_tab_input_changed(pmid: Optional[str]=None):
        if pmid is None:
            pmid = ss.get("id-html-tab-input")

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