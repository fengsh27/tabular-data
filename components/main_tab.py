import os
from typing import Optional, List, Dict, Callable
import streamlit as st
from datetime import datetime
import logging
from nanoid import generate
from streamlit_modal import Modal

import ast
import json
import time

from extractor.constants import ( 
    LLM_CHATGPT_4O,
    PROMPTS_NAME_PE,
    PROMPTS_NAME_PK,
    PROMPTS_NAME_PK_CHAIN,
    PROMPTS_NAME_PK_COT,
    LLM_CHATGPT_35,
    LLM_CHATGPT_40,
    LLM_GEMINI_FLASH,
    LLM_GEMINI_PRO,
)
from extractor.stampers import ArticleStamper, Stamper
from extractor.article_retriever import ExtendArticleRetriever, ArticleRetriever
from extractor.request_openai import (
    request_to_chatgpt_4o,
)
from extractor.utils import (
    convert_csv_table_to_dataframe,
    convert_html_table_to_dataframe,
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
    request_to_gemini_15_pro,
    request_to_gemini_15_flash,
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

def on_input_change(pmid: Optional[str]=None):
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
    ss.main_info = f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Retrieving completed, {tmp_info}"


def on_extract(pmid: str):
    global stamper

    # initialize
    pmid = pmid.strip()
    set_stamper_pmid(pmid)
    clear_results()

    if ss.main_prompts_option == PROMPTS_NAME_PK_CHAIN:
        include_tables = []
        for ix in range(len(ss.main_retrieved_tables)):
            include_tbl = ss.get(f"w-pmid-tbl-check-{ix}")
            if include_tbl:
                include_tables.append(ss.main_retrieved_tables[ix])
        if len(include_tables) > 0:
            shared_table_content = generate_tables_prompts(include_tables)

        customized_prompts = ss.main_customized_prompts
        if customized_prompts and len(customized_prompts) > 0:
            st.error("Customization is not permitted for prompt chaining.")
            return

        if len(include_tables) <= 0:
            st.error("Please select at least one table")
            return

        try:
            fobj = open("./prompts/chainprompts/pk_prompt_chain.json", "r")
        except Exception as e:
            logger.error(e)
            st.error(e)

        json_content = json.load(fobj)

        """ Step 1 - Identify PK Tables """
        """ Analyze the given HTML to determine which tables are about PK. """
        """ Example response: ["Table 1", "Table 2"] """

        step1_content: Optional[Dict] = json_content.get("step1", None)
        step1_seek_for_pk_table = step1_content.get('seek_for_pk_table', None)
        step1_format = step1_content.get('format', None)
        step1_prompt_list = [{"role": "user", "content": step1_seek_for_pk_table},
                             {"role": "user", "content": shared_table_content},
                             {"role": "user", "content": step1_format}]

        try:
            stamper.output_prompts(step1_prompt_list)
            request_llm: Optional[Callable[[List[Dict[str,str],str],str]]] = None
            if ss.main_llm_option == LLM_CHATGPT_4O:
                request_llm = request_to_chatgpt_4o
            else:
                request_llm = request_to_gemini_15_pro
            step1_res, step1_content, step1_usage, step1_truncated = request_llm(
                step1_prompt_list,
                step1_format,
            )
            start = step1_content.find("[")
            end = step1_content.rfind("]")
            if start != -1 and end != -1:
                step1_content = step1_content[start:end + 1]
            step1_content = ast.literal_eval(str(step1_content))  # Convert to a real Python list

            notification = "From the HTML you selected, the following table(s) are related to PK (Pharmacokinetics): "
            for tn in step1_content:
                notification += tn
                notification += ", "
            st.write(notification)
            st.write("Step 1 completed, token usage:", str(step1_usage))

        except Exception as e:
            logger.error(e)
            st.error(e)
            return

        """ Step 2 - Further Divide Each Table """
        """ Divide each PK table into subsections. """
        """ Example response: ["Overall", "3 Month to < 3 Years", "3 to < 13 Years", "13 to < 18 Years"] or simply False """
        # step1_content = ['Table II', 'Table III']

        table_sections = dict()  # table_name -> subsections

        time.sleep(0.1)

        step2_usage_list = []

        for table_name in step1_content:
            step2_content: Optional[Dict] = json_content.get("step2", None)
            step2_further_divide_the_table = step2_content.get('further_divide_the_table', None)
            step2_further_divide_the_table = step2_further_divide_the_table.replace('TABLE_NAME', table_name)
            step2_format = step2_content.get('format', None)
            step2_format = step2_format.replace('TABLE_NAME', table_name)
            step2_prompt_list = [{"role": "user", "content": step2_further_divide_the_table},
                                 {"role": "user", "content": shared_table_content},
                                 {"role": "user", "content": step2_format}]
            for attempt in range(5):
                try:
                    stamper.output_prompts(step2_prompt_list)
                    request_llm: Optional[Callable[[List[Dict[str,str],str],str]]] = None
                    if ss.main_llm_option == LLM_CHATGPT_4O:
                        request_llm = request_to_chatgpt_4o
                    else:
                        request_llm = request_to_gemini_15_pro
                    step2_res, step2_content, step2_usage, step2_truncated = request_llm(
                        step2_prompt_list,
                        step2_format,
                    )
                    step2_usage_list.append(step2_usage)
                    if '[' in step2_content and ']' in step2_content:
                        start = step2_content.find("[")
                        end = step2_content.rfind("]")
                        if start != -1 and end != -1:
                            step2_content = step2_content[start:end + 1]
                        step2_content = ast.literal_eval(str(step2_content))  # Convert to a real Python list
                        table_sections[table_name] = step2_content
                        notification = f"{table_name} can be split into the following sections: "
                        for sn in step2_content:
                            notification += sn
                            notification += ", "
                        st.write(notification)
                    # elif 'False' in step2_content:
                    else:
                        table_sections[table_name] = None
                        st.write(f"{table_name} cannot be further split.")
                    break
                except Exception as e:
                    logging.error(f"Split {table_name}, attempt {attempt + 1} failed: {e}")
                    st.error(f"Split {table_name}, attempt {attempt + 1} failed: {e}")
                    time.sleep(0.1 * (attempt + 1))
                    table_sections[table_name] = None

        st.write("Step 2 completed, token usage:", str(sum(step2_usage_list)))

        """ Step 3 - Extract from Each Table Section """
        """ Extract the information from each section of each table. """

        # table_sections = {'Table II': None, 'Table III': ["Overall", "3 Month to < 3 Years", "3 to < 13 Years", "13 to < 18 Years"]}
        # table_sections = {'Table II': None, 'Table III': ["Overall", "3 Month to < 3 Years"]}
        # table_sections = {'Table II': None}
        time.sleep(0.1)

        table_section_name_list = []

        for tn in table_sections.keys():
            if table_sections[tn] is None:
                table_section_name_list.append(tn)
            else:
                for sn in table_sections[tn]:
                    table_section_name_list.append(f"`{sn}` part of {tn}")

        # st.write(table_section_name_list)
        step3_usage_list = []
        final_csv_list = []

        for table_section_name in table_section_name_list:
            step3_content: Optional[Dict] = json_content.get("step3", None)
            step3_extract_from_each_table_section = step3_content.get('extract_from_each_table_section', None)
            step3_extract_from_each_table_section = step3_extract_from_each_table_section.replace('TABLE_SECTION_NAME', table_section_name)
            step3_format = step3_content.get('format', None)
            step3_format = step3_format.replace('TABLE_SECTION_NAME', table_section_name)
            step3_prompt_list = [
                {"role": "user", "content": step3_extract_from_each_table_section},
                {"role": "user", "content": shared_table_content},
                {"role": "user", "content": step3_format}]

            for attempt in range(5):
                try:
                    stamper.output_prompts(step3_prompt_list)
                    request_llm: Optional[Callable[[List[Dict[str, str]], str], str]] = None
                    if ss.main_llm_option == LLM_CHATGPT_4O:
                        request_llm = request_to_chatgpt_4o
                    else:
                        request_llm = request_to_gemini_15_pro
                    step3_res, step3_content, step3_usage, step3_truncated = request_llm(
                        step3_prompt_list,
                        step3_format,
                    )
                    step3_usage_list.append(step3_usage)
                    if '[' in step3_content and ']' in step3_content:
                        start = step3_content.find("[")
                        end = step3_content.rfind("]")
                        if start != -1 and end != -1:
                            st.write(f"Processing {table_section_name}")
                            processor = GeneratedPKSummaryTableProcessor(PROMPTS_NAME_PK)
                            has_header = table_section_name == table_section_name_list[0]
                            csv_str = processor.process_content(step3_content[start:end + 1], has_header)
                            # st.write(csv_str)
                            final_csv_list.append(csv_str)
                    break
                except Exception as e:
                    logging.error(f"{table_section_name}, attempt {attempt + 1} failed: {e}")
                    st.error(f"{table_section_name}, attempt {attempt + 1} failed: {e}")
                    time.sleep(0.1*(attempt + 1))
        final_csv_str = ''.join(final_csv_list)
        st.write("Step 3 completed, token usage:", str(sum(step3_usage_list)))

        ss.main_extracted_result = final_csv_str
        ss.main_token_usage = step1_usage + sum(step2_usage_list) + sum(step3_usage_list)
        # ss.main_token_usage = sum(step3_usage_list)
        return

    # prepare prompts including article prmpots and table prompts
    """
        PK CoT is modified based on the original PK prompt. 
        TableExtractionPromptsGenerator and GeneratedPKSummaryTableProcessor 
        are still initialized with PROMPTS_NAME_PK to minimize code changes.
    """
    prompt_option = PROMPTS_NAME_PK if ss.main_prompts_option == PROMPTS_NAME_PK_COT else ss.main_prompts_option
    prmpt_generator = TableExtractionPromptsGenerator(prompt_option)
    first_prompots = prmpt_generator.generate_system_prompts()

    if ss.main_prompts_option == PROMPTS_NAME_PK_COT:
        with open(f"./prompts/cot_examples/0119_shot29943508.txt", "r") as file:
            cot_example = file.read()
        first_prompots += cot_example

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
    # global stamper
    try:
        tmp_prmpts_list = [*prompts_list, {"role": "user", "content": generate_question(source)}]
        stamper.output_prompts(tmp_prmpts_list)
        request_llm: Optional[Callable[[List[Dict[str, str], str], str]]] = None
        if ss.main_llm_option == LLM_CHATGPT_4O:
            request_llm = request_to_chatgpt_4o
        else:
            request_llm = request_to_gemini_15_pro

        res, content, usage, truncated = request_llm(
            prompts_list,
            generate_question(source),
        )

        stamper.output_result(f"{content}\n\nUsage: {str(usage) if usage is not None else ''}")
        processor = GeneratedPKSummaryTableProcessor(prompt_option)
        csv_str = processor.process_content(content)
        ss.main_extracted_result = csv_str
        ss.main_token_usage = usage

        ss.main_info = f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} Extracting completed. {'***The result includes truncated contents.***' if truncated is True else ''}"
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
    ss.setdefault("main_llm_option", LLM_CHATGPT_40)
    ss.setdefault("main_customized_prompts", "")

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
            usage = ss.main_token_usage
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
            LLM_CHATGPT_4O, LLM_GEMINI_PRO
        ), index=0)
        ss.main_llm_option = llm_option
        st.divider()
        prompts_array = (PROMPTS_NAME_PK, PROMPTS_NAME_PE, PROMPTS_NAME_PK_COT, PROMPTS_NAME_PK_CHAIN)
        option = st.selectbox("What type of prompts would you like to use?", prompts_array, index=0)
        ss.main_prompts_option = option
        open_modal = st.button("View Prompts ...")
        st.divider()
        customized_prompts = st.text_area("Customized Prompts (Optional):", height=70)
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
        generator = TableExtractionPromptsGenerator(ss.main_prompts_option)
        prmpts = generator.get_prompts_file_content()
        prmpts += "\n\n\n"
        with modal.container():
            st.text(prmpts)
            st.divider()
