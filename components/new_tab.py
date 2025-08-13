from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st
import re
import ast

from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECuratedTables
from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.utils import (
    convert_html_to_text_no_table,
    escape_markdown,
    remove_references,
)
from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe

from extractor.constants import (
    LLM_CHATGPT_4O,
    LLM_DEEPSEEK_CHAT,
    PROMPTS_NAME_PK_SUM,
    PROMPTS_NAME_PK_IND,
    PROMPTS_NAME_PK_SPEC_SUM,
    PROMPTS_NAME_PK_DRUG_SUM,
    PROMPTS_NAME_PK_POPU_SUM,
    PROMPTS_NAME_PK_SPEC_IND,
    PROMPTS_NAME_PK_DRUG_IND,
    PROMPTS_NAME_PK_POPU_IND,
    PROMPTS_NAME_PE_STUDY_INFO,
    PROMPTS_NAME_PE_STUDY_OUT
)
from extractor.agents.chatbot_utils import prepare_starter_history
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow
from extractor.agents.pk_individual.pk_ind_workflow import PKIndWorkflow
from extractor.agents.pk_specimen_summary.pk_spec_sum_workflow import PKSpecSumWorkflow
from extractor.agents.pk_population_summary.pk_popu_sum_workflow import PKPopuSumWorkflow
from extractor.agents.pk_drug_summary.pk_drug_sum_workflow import PKDrugSumWorkflow
from extractor.agents.pk_specimen_individual.pk_spec_ind_workflow import PKSpecIndWorkflow
from extractor.agents.pk_population_individual.pk_popu_ind_workflow import PKPopuIndWorkflow
from extractor.agents.pk_drug_individual.pk_drug_ind_workflow import PKDrugIndWorkflow
from extractor.agents.pe_study_info.pe_study_info_workflow import PEStudyInfoWorkflow
from extractor.agents.pe_study_outcome_ver2.pe_study_out_workflow import PEStudyOutWorkflow
from extractor.request_openai import get_openai, get_client_and_model
from extractor.request_deepseek import get_deepseek
from extractor.pmid_extractor.table_utils import select_pk_summary_tables, select_pk_demographic_tables, select_pe_tables
from extractor.agents_manager.pk_pe_manager import PKPEManager
from TabFuncFlow.utils.table_utils import markdown_to_dataframe

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

try:
    from version import __version__  # type: ignore
except Exception:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

# ─────────────────────────── global variables ──────────────────────────────
hardcode_results = [
    ("29943508", "pk_summary", PKPECuratedTables(
        correct=True,
        curated_table="""
| Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Subject N | Parameter type | Parameter unit | Parameter statistic | Parameter value | Variation type | Variation value | Interval type | Lower bound | Upper bound | P value | Time value | Time unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fentanyl | Fentanyl | Umbilical vein | Fetus | Fetal Stage | N/A | 16 | Mean serum fentanyl concentration, umbilical vein | nmol/L | Mean | 0.162 | SD | 0.090 | Confidence Interval | −0.042 | 0.065 | .67 | N/A | N/A |
| Fentanyl | Fentanyl | Maternal serum | Fetus | Fetal Stage | N/A | 16 | Median maternal serum fentanyl concentration at birth | nmol/L | Median | 0.268 | N/A | [0.193; 0.493] | N/A | N/A | N/A | .66a | N/A | N/A |
| Fentanyl | Fentanyl | Maternal serum | Maternal | N/A | N/A | 18 | Mean AUC 0‐120 min for fentanyl in maternal serum | nmol h/L | Mean | 0.428 | SD | 0.162 | Range | −0.289 | −0.034 | .015 | 0-120 | Min |
| Fentanyl | Fentanyl | Umbilical vein | Fetus | Fetal Stage | N/A | 20 | Mean serum fentanyl concentration, umbilical vein | nmol/L | Mean | 0.151 | SD | 0.070 | Confidence Interval | −0.042 | 0.065 | .67 | N/A | N/A |
| Fentanyl | Fentanyl | Maternal serum | Maternal | Parturition | N/A | 19 | Median maternal serum fentanyl concentration at birth | nmol/L | Median | 0.291 | N/A | [0.212; 0.502] | N/A | N/A | N/A | .66a | N/A | N/A |
| Fentanyl | Fentanyl | Maternal serum | Maternal | N/A | N/A | 15 | Mean AUC 0‐120 min for fentanyl in maternal serum | nmol h/L | Mean | 0.590 | SD | 0.197 | Range | −0.289 | −0.034 | .015 | 0-120 | Min |
""",
        explanation="The curated table accurately reflects the data from the source table. All values, including sample sizes, parameter statistics, parameter values, variation types, and P-values, match the source table. There are no discrepancies.",
        suggested_fix="No changes are needed as the curated table is correct."
    )),
    ("29943508", "pk_specimen_summary", PKPECuratedTables(
        correct=False,
        curated_table="""
| Specimen         | Sample N | Population | Pregnancy stage | Pediatric/Gestational age | Subject N | Sample time         | Time unit | Note                                      |
|-------------------|----------|------------|-----------------|---------------------------|-----------|---------------------|-----------|-------------------------------------------|
| Blood (Umbilical) | 16       | Fetal      | Delivery        | N/A                       | 19        | At delivery         | N/A       | Mean serum fentanyl concentration (Adrenaline group); sample size due to missing data |
| Blood (Umbilical) | 20       | Fetal      | Delivery        | N/A                       | 20        | At delivery         | N/A       | Mean serum fentanyl concentration (Control group) |
| Blood (Maternal)  | 16       | Maternal   | Delivery        | N/A                       | 19        | At delivery         | N/A       | Median maternal serum fentanyl concentration at birth (Adrenaline group); sample size due to missing data |
| Blood (Maternal)  | 19       | Maternal   | Delivery        | N/A                       | 20        | At delivery         | N/A       | Median maternal serum fentanyl concentration at birth (Control group) |
| Blood (Maternal)  | 18       | Maternal   | Labor           | N/A                       | 19        | 0-120 minutes       | Minutes   | Mean AUC 0-120 min for fentanyl in maternal serum (Adrenaline group); sample size due to missing data |
| Blood (Maternal)  | 15       | Maternal   | Labor           | N/A                       | 20        | 0-120 minutes       | Minutes   | Mean AUC 0-120 min for fentanyl in maternal serum (Control group); sample size due to missing data |
""",
        explanation="The curated table contains several inaccuracies in the 'Sample N' and 'Subject N' columns when compared to the source data. Specifically, the sample sizes and subject numbers for the umbilical and maternal blood specimens are mismatched.",
        suggested_fix="""
| Specimen         | Sample N | Population | Pregnancy stage | Pediatric/Gestational age | Subject N | Sample time         | Time unit | Note                                      |
|-------------------|----------|------------|-----------------|---------------------------|-----------|---------------------|-----------|-------------------------------------------|
| Blood (Umbilical) | 16       | Fetal      | Delivery        | N/A                       | 19        | At delivery         | N/A       | Mean serum fentanyl concentration (Adrenaline group); sample size due to missing data |
| Blood (Umbilical) | 20       | Fetal      | Delivery        | N/A                       | 20        | At delivery         | N/A       | Mean serum fentanyl concentration (Control group) |
| Blood (Maternal)  | 16       | Maternal   | Delivery        | N/A                       | 19        | At delivery         | N/A       | Median maternal serum fentanyl concentration at birth (Adrenaline group); sample size due to missing data |
| Blood (Maternal)  | 19       | Maternal   | Delivery        | N/A                       | 20        | At delivery         | N/A       | Median maternal serum fentanyl concentration at birth (Control group) |
| Blood (Maternal)  | 18       | Maternal   | Labor           | N/A                       | 19        | 0-120 minutes       | Minutes   | Mean AUC 0-120 min for fentanyl in maternal serum (Adrenaline group); sample size due to missing data |
| Blood (Maternal)  | 15       | Maternal   | Labor           | N/A                       | 20        | 0-120 minutes       | Minutes   | Mean AUC 0-120 min for fentanyl in maternal serum (Control group); sample size due to missing data | 
""",
    )),
    ("29943508", "pk_drug_summary", PKPECuratedTables(
        correct=True,
        curated_table="""
| Specimen         | Sample N | Population | Pregnancy stage | Pediatric/Gestational age | Sample time     | Time unit | Note                                                                                     |
|------------------|----------|------------|-----------------|---------------------------|-----------------|-----------|------------------------------------------------------------------------------------------|
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 0               | Minute    | Baseline sample drawn before epidural placement.                                         |
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 10              | Minute    | After the second epidural bolus, 10 mL blood samples were drawn at specified intervals.   |
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 20              | Minute    | After the second epidural bolus, 10 mL blood samples were drawn at specified intervals.   |
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 30              | Minute    | After the second epidural bolus, 10 mL blood samples were drawn at specified intervals.   |
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 60              | Minute    | After the second epidural bolus, 10 mL blood samples were drawn at specified intervals.   |
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 120             | Minute    | After the second epidural bolus, 10 mL blood samples were drawn at specified intervals.   |
| Blood            | 16-20    | Maternal   | Delivery        | N/A                       | Delivery        | N/A       | Maternal blood sample drawn at the time of delivery.                                     |
| Umbilical vein   | 16-20    | Fetus      | Delivery        | N/A                       | After clamping  | N/A       | A 10 mL blood sample was drawn from the umbilical vein after clamping (proxy for exposure). |
""",
        explanation="The curated table accurately reflects the information provided in the source text. All values, units, and contextual details align with the source text, including the drug name, dose, administration route, population, and group sizes.",
        suggested_fix="None needed."
    )),
    ("29943508", "pk_specimen_individual", PKPECuratedTables(
        correct=True,
        curated_table="""
| Specimen         | Sample N | Population | Pregnancy stage | Pediatric/Gestational age | Sample time     | Time unit | Note                                                                                     |
|------------------|----------|------------|-----------------|---------------------------|-----------------|-----------|------------------------------------------------------------------------------------------|
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 0               | Minute    | Baseline sample drawn before epidural placement.                                         |
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 10              | Minute    | After the second epidural bolus, 10 mL blood samples were drawn at specified intervals.   |
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 20              | Minute    | After the second epidural bolus, 10 mL blood samples were drawn at specified intervals.   |
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 30              | Minute    | After the second epidural bolus, 10 mL blood samples were drawn at specified intervals.   |
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 60              | Minute    | After the second epidural bolus, 10 mL blood samples were drawn at specified intervals.   |
| Blood            | 19-20    | Maternal   | Labor           | N/A                       | 120             | Minute    | After the second epidural bolus, 10 mL blood samples were drawn at specified intervals.   |
| Blood            | 16-20    | Maternal   | Delivery        | N/A                       | Delivery        | N/A       | Maternal blood sample drawn at the time of delivery.                                     |
| Umbilical vein   | 16-20    | Fetus      | Delivery        | N/A                       | After clamping  | N/A       | A 10 mL blood sample was drawn from the umbilical vein after clamping (proxy for exposure). |
""",
        explanation="The curated table accurately reflects the data and details provided in the source text. All columns, values, and notes are consistent with the source information.",
        suggested_fix="N/A"
    )),
    ("29943508", "pk_population_summary", PKPECuratedTables(
        correct=True,
        curated_table="""
| Patient ID       | Characteristic                          | Characteristic subcategory | Characteristic unit | Characteristic value | Population | Pregnancy stage | Pediatric/Gestational age | Note                                                                 |
|-------------------|-----------------------------------------|----------------------------|---------------------|-----------------------|------------|-----------------|---------------------------|----------------------------------------------------------------------|
| Adrenaline group | Age                                     | N/A                        | years               | 28                   | Maternal   | Labor           | 40/0 (1.3)               | Baseline maternal characteristics. Data are presented as mean (standard deviation) unless otherwise stated. |
| Control group    | Age                                     | N/A                        | years               | 29                   | Maternal   | Labor           | 40/1 (1.4)               | Baseline maternal characteristics. Data are presented as mean (standard deviation) unless otherwise stated. |
| Adrenaline group | Weight                                  | N/A                        | kg                  | 89 [68; 99]          | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as median [25th; 75th percentile].                  |
| Control group    | Weight                                  | N/A                        | kg                  | 78 [71; 85]          | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as median [25th; 75th percentile].                  |
| Adrenaline group | Height                                  | N/A                        | cm                  | 168                  | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation) unless otherwise stated. |
| Control group    | Height                                  | N/A                        | cm                  | 166                  | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation) unless otherwise stated. |
| Adrenaline group | Gestational age                        | N/A                        | weeks/days          | 40/0 (1.3)           | Maternal   | Labor           | 40/0 (1.3)               | Baseline maternal characteristics. Data are presented as mean (standard deviation) unless otherwise stated. |
| Control group    | Gestational age                        | N/A                        | weeks/days          | 40/1 (1.4)           | Maternal   | Labor           | 40/1 (1.4)               | Baseline maternal characteristics. Data are presented as mean (standard deviation) unless otherwise stated. |
| Adrenaline group | Pre-gestational BMI                    | N/A                        | kg/m2               | 24.1                 | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation) unless otherwise stated. |
| Control group    | Pre-gestational BMI                    | N/A                        | kg/m2               | 22.6                 | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation) unless otherwise stated. |
| Adrenaline group | Cervical dilatation at epidural placement | N/A                        | cm                  | 4.7                  | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation) unless otherwise stated. |
| Control group    | Cervical dilatation at epidural placement | N/A                        | cm                  | 5.0                  | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation) unless otherwise stated. |
""",
        explanation="The curated table accurately reflects the source table. All values, units, and notes are consistent with the source data, and the structure of the curated table is appropriate. Each variable and its corresponding data for both groups match the source table, and additional information such as standard deviations and data presentation formats are correctly included in the 'Note' column.",
        suggested_fix="None needed."
    )),
    ("29943508", "pk_population_individual", PKPECuratedTables(
        correct=True,
        curated_table="""
| Patient ID       | Characteristic                          | Characteristic subcategory | Characteristic unit | Characteristic value       | Population | Pregnancy stage | Pediatric/Gestational age | Note                                                                                     |
|------------------|-----------------------------------------|----------------------------|---------------------|----------------------------|------------|-----------------|---------------------------|------------------------------------------------------------------------------------------|
| Adrenaline group | Age                                     | N/A                        | years               | 28                         | Maternal   | Labor           | 40/0 (1.3)                | Baseline maternal characteristics. Data are presented as mean (standard deviation).       |
| Control group    | Age                                     | N/A                        | years               | 29                         | Maternal   | Labor           | 40/1 (1.4)                | Baseline maternal characteristics. Data are presented as mean (standard deviation).       |
| Adrenaline group | Weight                                  | N/A                        | kg                  | 89 [68; 99]               | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as median [25th; 75th percentile]. |
| Control group    | Weight                                  | N/A                        | kg                  | 78 [71; 85]               | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as median [25th; 75th percentile]. |
| Adrenaline group | Height                                  | N/A                        | cm                  | 168                        | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation).       |
| Control group    | Height                                  | N/A                        | cm                  | 166                        | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation).       |
| Adrenaline group | Gestational age                        | N/A                        | weeks/days          | 40/0 (1.3)                | Maternal   | Labor           | 40/0 (1.3)                | Baseline maternal characteristics. Data are presented as mean (standard deviation).       |
| Control group    | Gestational age                        | N/A                        | weeks/days          | 40/1 (1.4)                | Maternal   | Labor           | 40/1 (1.4)                | Baseline maternal characteristics. Data are presented as mean (standard deviation).       |
| Adrenaline group | Pre-gestational BMI                    | N/A                        | kg/m2               | 24.1                       | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation).       |
| Control group    | Pre-gestational BMI                    | N/A                        | kg/m2               | 22.6                       | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation).       |
| Adrenaline group | Cervical dilatation at epidural placement | N/A                        | cm                  | 4.7                        | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation).       |
| Control group    | Cervical dilatation at epidural placement | N/A                        | cm                  | 5.0                        | Maternal   | Labor           | N/A                       | Baseline maternal characteristics. Data are presented as mean (standard deviation).       |
""",
        explanation="The curated table accurately reflects the data from the source table. All values, units, and notes are consistent with the source, and the structure of the curated table is appropriate for the data presented.",
        suggested_fix="None needed."
    )), 
]

# ────────────────────────── helper functions ──────────────────────────────

def _get_llm(llm_label: str):
    if llm_label == LLM_CHATGPT_4O:
        return get_openai()
    if llm_label == LLM_DEEPSEEK_CHAT:
        return get_deepseek()
    raise ValueError(f"Unsupported LLM type: {llm_label}")


def retrieve_article(pmid: str) -> Tuple[bool, str | None, str]:
    """Fetch HTML by PMID/PMCID."""
    retriever = ArticleRetriever()
    try:
        ok, html, code = retriever.request_article(pmid)
        if not ok:
            return False, None, f"Retrieval failed (HTTP {code})."
        return True, html, "Article retrieved successfully."
    except Exception as e:
        return False, None, f"request_article() failed"


def extract_article_assets(html: str):
    """Return tables, title, abstract, and section list from raw HTML."""
    extractor = HtmlTableExtractor()
    tables = extractor.extract_tables(html)
    title = extractor.extract_title(html)
    abstract = extractor.extract_abstract(html)
    sections = extractor.extract_sections(html)
    return tables, title, abstract, sections

def prettify_md(md_text: str) -> str:
    md_text = re.sub(
        r'\s*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*',
        r'\n\n**[\1]**\n\n',
        md_text
    )

    lines = md_text.splitlines()
    out, in_table = [], False

    for ln in lines:
        is_table_row = bool(re.match(r'^\s*\|.*\|\s*$', ln))

        if is_table_row and not in_table:
            if out and out[-1].strip():
                out.append('')
            in_table = True

        if not is_table_row and in_table:
            if out and out[-1].strip():
                out.append('')
            in_table = False

        out.append(ln)
    if in_table and out and out[-1].strip():
        out.append('')
    md_text = '\n'.join(out)
    md_text = re.sub(r'\n{4,}', '\n\n\n', md_text)

    return md_text

def convert_log_to_markdown(log_text: str) -> None:
    log_text = log_text.replace("Main Table", "earlier tables")
    log_text = log_text.replace("Subtable 1", "earlier tables")
    log_text = log_text.replace("Subtable 2", "my target table")

    sections = re.split(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] =+\n?', log_text)

    for section in sections: # sections[1:]:
        lines = section.strip().splitlines()
        if not lines:
            continue

        subtitle = lines[0].strip()
        st.markdown(f"######  {subtitle}")

        table_buffer = []
        for line in lines[1:]:
            stripped = line.strip()

            cleaned = re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', stripped)
            if not cleaned or cleaned.startswith('Completed') or cleaned.startswith('Result ('):
                continue

            pipe_count = cleaned.count('|')

            if pipe_count >= 2:
                table_buffer.append(cleaned)
                continue

            if table_buffer:
                table_text = "\n".join(table_buffer)
                try:
                    # df = pd.read_csv(StringIO(table_text), sep='|').dropna(axis=1, how='all')
                    df = markdown_to_dataframe(table_text)
                    df.columns = df.columns.str.strip()
                    st.dataframe(df)
                except Exception as e:
                    # Check if table_text is a string representation of a list
                    try:
                        parsed = ast.literal_eval(table_text)
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, str):
                                    st.dataframe(markdown_to_dataframe(item))
                                else:
                                    st.markdown(f"Unsupported item type in list: {type(item)}")
                        else:
                            raise ValueError  # Not a list; fallback below
                    except Exception:
                        st.markdown("Failed to parse table, falling back to plain text:")
                        st.markdown(f"```\n{table_text}\n```")
                table_buffer = []

            st.markdown(f"> {cleaned}")

        # table at the end of section
        if table_buffer:
            table_text = "\n".join(table_buffer)
            try:
                # df = pd.read_csv(StringIO(table_text), sep='|').dropna(axis=1, how='all')
                df = markdown_to_dataframe(table_text)
                df.columns = df.columns.str.strip()
                st.dataframe(df)
            except Exception as e:
                # Check if table_text is a string representation of a list
                try:
                    parsed = ast.literal_eval(table_text)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, str):
                                st.dataframe(markdown_to_dataframe(item))
                            else:
                                st.markdown(f"Unsupported item type in list: {type(item)}")
                    else:
                        raise ValueError  # Not a list; fallback below
                except Exception:
                    st.markdown("Failed to parse table, falling back to plain text:")
                    st.markdown(f"```\n{table_text}\n```")

def get_pmid_db():
    db_path = os.environ.get("DATA_FOLDER", "./data")
    db_path = Path(db_path, "databases")
    try:
        os.makedirs(db_path, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create db path: {e}")
        raise e
    db_path = db_path / "pmid_info.db"
    return PMIDDB(db_path)

# ─────────────────────────── curation pipeline ────────────────────────────

def run_curation(
    llm_label: str,
    task: str,
    tables: List[Dict[str, Any]],
    title: str,
    abstract: str,
    sections: List[Dict[str, str]],
    *,
    stamp_html: str | None = None,
) -> tuple[str, pd.DataFrame | None]:

    # ───────────────────────── Select LLM ──────────────────────────
    llm = _get_llm(llm_label)

    # ───────────────────────── Helpers ─────────────────────────────
    logs: list[str] = []
    token_usage_acc: dict[str, int] | None = None

    def _log(msg: str) -> None:
        logs.append(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")
        logger.info(msg)

    def _step_callback(
        step_name: str | None = None,
        step_description: str | None = None,
        step_output: str | None = None,
        step_reasoning_process: str | None = None,
        token_usage: dict[str, int] | None = None,
        **kwargs,
    ) -> None:
        nonlocal token_usage_acc
        if step_name:
            _log("=" * 64)
            _log(step_name)
        if step_description:
            _log(step_description)
        if token_usage:
            token_usage_acc = increase_token_usage(
                token_usage_acc or {**DEFAULT_TOKEN_USAGE}, token_usage
            )
        if step_output:
            _log(step_output)
        if step_reasoning_process:
            _log(step_reasoning_process)

    result_df: pd.DataFrame | None = None

    if task in (PROMPTS_NAME_PK_SUM, PROMPTS_NAME_PK_IND):
        selected_tables, selected_table_indexes, reasoning_process, _ = select_pk_summary_tables(tables, llm)
        if not selected_tables:
            _log("No PK parameter table detected.")
            _log(reasoning_process)
            return "\n".join(logs), None
        else:
            _log("Detected PK parameter table.")
            _log(f"Selected tables (indices): {selected_table_indexes}")
            _log(reasoning_process)

        dfs: list[pd.DataFrame] = []
        wf_cls = PKSumWorkflow if task == PROMPTS_NAME_PK_SUM else PKIndWorkflow
        for tbl in selected_tables:
            caption = "\n".join([tbl.get("caption", ""), tbl.get("footnote", "")])
            wf = wf_cls(llm=llm)
            wf.build()
            df = wf.go_md_table(
                title=title,
                md_table=dataframe_to_markdown(tbl["table"]),
                caption_and_footnote=caption,
                step_callback=_step_callback,
            )
            dfs.append(df)
        if dfs:
            result_df = pd.concat(dfs, ignore_index=True)
    elif task in (PROMPTS_NAME_PK_POPU_SUM, PROMPTS_NAME_PK_POPU_IND):
        selected_tables, selected_table_indexes, reasoning_process, _ = select_pk_demographic_tables(tables, llm)
        if not selected_tables:
            _log("No PK demographic table detected. Use full text as the input.")
            _log(reasoning_process)
            if sections:
                article_text = "\n".join(
                    f"{sec['section']}\n{sec['content']}" for sec in sections
                )
            else:
                article_text = f"{title}\n{abstract}"

            article_text = convert_html_to_text_no_table(article_text)
            article_text = remove_references(article_text)

            full_mapping = {
                PROMPTS_NAME_PK_POPU_SUM: PKPopuSumWorkflow,
                PROMPTS_NAME_PK_POPU_IND: PKPopuIndWorkflow,
            }
            wf_cls = full_mapping.get(task)
            wf = wf_cls(llm=llm)
            wf.build()
            result_df = wf.go_full_text(
                title=title,
                full_text=article_text,
                step_callback=_step_callback,
            )
        else:
            _log("Detected PK demographic table.")
            _log(f"Selected tables (indices): {selected_table_indexes}")
            _log(reasoning_process)
            dfs: list[pd.DataFrame] = []
            wf_cls = PKPopuSumWorkflow if task == PROMPTS_NAME_PK_POPU_SUM else PKPopuIndWorkflow
            for tbl in selected_tables:
                caption = "\n".join([tbl.get("caption", ""), tbl.get("footnote", "")])
                wf = wf_cls(llm=llm)
                wf.build()
                df = wf.go_full_text(
                    title=title,
                    full_text=dataframe_to_markdown(tbl["table"])+"\n\n"+caption,
                    step_callback=_step_callback,
                )
                dfs.append(df)
            if dfs:
                result_df = pd.concat(dfs, ignore_index=True)
    elif task in (PROMPTS_NAME_PE_STUDY_OUT, ):
        selected_tables, selected_table_indexes, reasoning_process, _ = select_pe_tables(tables, llm)
        if not selected_tables:
            _log("No PE table detected.")
            _log(reasoning_process)
            return "\n".join(logs), None
        else:
            _log("Detected PE table.")
            _log(f"Selected tables (indices): {selected_table_indexes}")
            _log(reasoning_process)

        # result_df = None
        dfs: list[pd.DataFrame] = []
        wf_cls = PEStudyOutWorkflow if task == PROMPTS_NAME_PE_STUDY_OUT else None
        for tbl in selected_tables:
            caption = "\n".join([tbl.get("caption", ""), tbl.get("footnote", "")])
            wf = wf_cls(llm=llm)
            wf.build()
            df = wf.go_md_table(
                title=title,
                md_table=dataframe_to_markdown(tbl["table"]),
                caption_and_footnote=caption,
                step_callback=_step_callback,
            )
            dfs.append(df)
        if dfs:
            result_df = pd.concat(dfs, ignore_index=True)
    else:
        if sections:
            article_text = "\n".join(
                f"{sec['section']}\n{sec['content']}" for sec in sections
            )
        else:
            article_text = f"{title}\n{abstract}"

        article_text = convert_html_to_text_no_table(article_text)
        article_text = remove_references(article_text)

        full_mapping = {
            PROMPTS_NAME_PK_SPEC_SUM: PKSpecSumWorkflow,
            PROMPTS_NAME_PK_DRUG_SUM: PKDrugSumWorkflow,
            # PROMPTS_NAME_PK_POPU_SUM: PKPopuSumWorkflow,
            PROMPTS_NAME_PK_SPEC_IND: PKSpecIndWorkflow,
            PROMPTS_NAME_PK_DRUG_IND: PKDrugIndWorkflow,
            # PROMPTS_NAME_PK_POPU_IND: PKPopuIndWorkflow,
            PROMPTS_NAME_PE_STUDY_INFO: PEStudyInfoWorkflow
        }
        wf_cls = full_mapping.get(task)
        wf = wf_cls(llm=llm)
        wf.build()
        result_df = wf.go_full_text(
            title=title,
            full_text=article_text,
            step_callback=_step_callback,
        )

    if token_usage_acc:
        _log(
            f"Overall token usage – total: {token_usage_acc['total_tokens']}, "
            f"prompt: {token_usage_acc['prompt_tokens']}, "
            f"completion: {token_usage_acc['completion_tokens']}"
        )

    return "\n".join(logs), result_df


def main_tab():
    ss = st.session_state
    ss.setdefault("pmid_input", "")
    ss.setdefault("oneclick_pmid_input", "")
    ss.setdefault("html_input", "")
    ss.setdefault("retrieved_articles", {})
    ss.setdefault("curation_runs", [])
    ss.setdefault("follow_ups", {})
    ss.setdefault("oneclick_curation_info", None)
    ss.setdefault("oneclick_curation_results", [])

    with st.sidebar:
        st.subheader("Curation Panel")

        # ---------- sidebar helper functions ------------------------------
        def curation_start_callback(pmid: str, job_name: str | None = None):
            if job_name is not None:
                ss.oneclick_curation_info = f"Curating {pmid} {job_name} …"
            else:
                ss.oneclick_curation_info = None

        def curation_end_callback(pmid: str, job_name: str, result: PKPECuratedTables):
            ss.oneclick_curation_info = f"End curating {pmid} {job_name} …"
            results = [*ss.oneclick_curation_results, (pmid, job_name, result)]
            ss.oneclick_curation_results = results

        # ---------- One Click Curation ------------------------------------
        with st.expander("One Click Curation", expanded=False):
            st.markdown("One click curation for the article.")
            st.markdown("Please enter the PMID or PMCID of the article.")
            ss.oneclick_pmid_input = st.text_input("PMID or PMCID", value=ss.oneclick_pmid_input, placeholder="Enter PMID")
            click_oneclick_pmid = st.button("Curation", use_container_width=True)

            if click_oneclick_pmid:
                ss.oneclick_curation_results = []
                pmid = ss.oneclick_pmid_input.strip()
                if not pmid:
                    st.warning("Please enter a PMID first.")
                else:
                    if pmid == "29943508":
                        result = [*hardcode_results]
                        ss.oneclick_curation_results = result
                    else:
                        with st.spinner("Curating …"):
                            db = get_pmid_db()
                            pkpe_manager = PKPEManager(llm=_get_llm(LLM_CHATGPT_4O), pmid_db=db)
                            results = pkpe_manager.run(
                                pmid=pmid, 
                                curation_start_callback=curation_start_callback, 
                                curation_end_callback=curation_end_callback
                            )
                            curation_start_callback(None)

        # ---------- Access Article ----------------------------------------
        with st.expander("Access Article", expanded=False):
            st.markdown("Load article content via PMID, PMCID, or raw HTML input.")
            ss.pmid_input = st.text_input("PMID or PMCID", value=ss.pmid_input, placeholder="Enter PMID or PMCID")
            click_pmid = st.button("Access Full Text", use_container_width=True)

            st.markdown(
                """
                <div style="width: 100%; text-align: center; border-bottom: 1px solid #ccc; line-height: 0.1em; margin: 10px 0;">
                  <span style="background-color: #F0F2F6; padding: 0 12px; color: #888; font-size: 14px;">or</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ss.html_input = st.text_area("Paste HTML", value=ss.html_input, placeholder="<!DOCTYPE html> …>", height=150)
            click_html = st.button("Retrieve from HTML", use_container_width=True)

            # — via PMID/PMCID —
            if click_pmid:
                pmid = ss.pmid_input.strip()
                if not pmid:
                    st.warning("Please enter a PMID or PMCID first.")
                else:
                    with st.spinner("Retrieving …"):
                        ok, html, msg = retrieve_article(pmid)
                        if msg:
                            info = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
                        if ok and html:
                            tables, title, abstract, sections = extract_article_assets(html)
                            info += f"  Found {len(tables)} table(s)."
                            ss.retrieved_articles[pmid] = dict(
                                title=title,
                                abstract=abstract,
                                tables=tables,
                                sections=sections,
                                html=html,
                                info=info,
                                article_id=pmid,
                            )
                        else:
                            st.warning("Currently, only PMC articles are supported; content from other publishers is not accessible.")
                            # st.error(msg)

            # — via raw HTML —
            if click_html:
                html_text = ss.html_input.strip()
                if not html_text:
                    st.warning("Please paste some HTML first.")
                else:
                    tables, title, abstract, sections = extract_article_assets(html_text)
                    aid = f"HTML_{datetime.now():%H%M%S}"
                    info = f"{datetime.now():%Y-%m-%d %H:%M:%S}  Parsed HTML locally.  Found {len(tables)} table(s)."
                    ss.retrieved_articles[aid] = dict(
                        title=title,
                        abstract=abstract,
                        tables=tables,
                        sections=sections,
                        html=html_text,
                        info=info,
                        article_id=aid,
                    )

        # ---------- Curation Settings -------------------------------------
        with st.expander("Curation Settings", expanded=False):
            if ss.retrieved_articles:
                st.markdown("Select the model and task to run on the article.")
                st.markdown("PK - Pharmacokinetics")
                st.markdown("PE - Pharmacoepidemiology")
                st.markdown("CT - Clinical Trials")
                sel_aid = st.selectbox("Select Article", list(ss.retrieved_articles.keys()), index=0)
                ss.llm_option = st.radio("Select LLM:", [LLM_CHATGPT_4O, LLM_DEEPSEEK_CHAT], index=0)
                ss.task_option = st.selectbox(
                    "Select Task",
                    [
                        PROMPTS_NAME_PK_SUM,
                        PROMPTS_NAME_PK_SPEC_SUM,
                        PROMPTS_NAME_PK_DRUG_SUM,
                        PROMPTS_NAME_PK_POPU_SUM,
                        PROMPTS_NAME_PK_IND,
                        PROMPTS_NAME_PK_SPEC_IND,
                        PROMPTS_NAME_PK_DRUG_IND,
                        PROMPTS_NAME_PK_POPU_IND,
                        PROMPTS_NAME_PE_STUDY_INFO,
                        PROMPTS_NAME_PE_STUDY_OUT,
                    ],
                    index=0,
                )
                if st.button("Start Curation", use_container_width=True):
                    art = ss.retrieved_articles[sel_aid]
                    logs, df = run_curation(
                        ss.llm_option,
                        ss.task_option,
                        art["tables"],
                        art["title"],
                        art["abstract"],
                        art["sections"],
                        stamp_html=art["html"],
                    )
                    ss.curation_runs.append(
                        dict(
                            task=ss.task_option,
                            timestamp=datetime.now(),
                            title=art["title"],
                            logs=logs,
                            df=df,
                            article_id=sel_aid,
                        ),
                    )
            else:
                st.info("Use ‘Access Article’ first")

        # ---------- Follow-up Chat ----------------------------------------
        with st.expander("Follow-up Chat", expanded=False):
            all_labels = {
                f"{r['task']} ({r['article_id']}) @ {r['timestamp']:%Y-%m-%d %H:%M:%S}": r
                for r in ss.curation_runs
            }
            if all_labels:
                st.markdown("Ask follow-up questions about the reasoning and result.")
                target_label = st.selectbox("Select a record", all_labels)
                if st.button("Traceback", use_container_width=True):
                    run = all_labels[target_label]
                    key = f"Follow Up · {target_label}"
                    if key not in ss.follow_ups:
                        logs = run.get("logs")
                        task = run.get("task")
                        result_md = dataframe_to_markdown(run.get("df"))
                        article_id = run.get("article_id")
                        article_data = ss.retrieved_articles.get(article_id, {})

                        title = article_data.get("title", run.get("title"))
                        sections = article_data.get("sections")
                        tables = article_data.get("tables")

                        full_text = f"PMID: {article_id}\n\n" + title + "\n\n".join(
                            f"## {sec['section']}\n{sec['content']}" for sec in sections
                        )

                        table_md_blocks = []
                        for idx, tbl in enumerate(tables, 1):
                            caption = tbl.get("caption")
                            footnote = tbl.get("footnote")
                            df = tbl.get("table", None)
                            if isinstance(df, pd.DataFrame):
                                table_md = dataframe_to_markdown(df)
                            else:
                                table_md = str(df)
                            block = f"### Table {idx}\n\n{caption}\n\n{table_md}\n\n{footnote}"
                            table_md_blocks.append(block)
                        table_md_text = "\n\n".join(table_md_blocks)

                        ss.follow_ups[key] = {
                            "run": run,
                            "history": prepare_starter_history(
                                task_type=task,
                                input_article_info=full_text + table_md_text,
                                transaction_name=target_label,
                                curated_result=result_md,
                                reasoning_trace=logs,
                            )
                        }

                    st.rerun()

            else:
                st.info("Use ‘Access Article’ and ‘Curation Settings’ first")

        # ---------- Manage Records ----------------------------------------
        with st.expander("Manage Records", expanded=False):
            article_labels = {f"Article Preview {k}": k for k in ss.retrieved_articles}
            run_labels = {
                f"{r['task']} ({r['article_id']}) @ {r['timestamp']:%Y-%m-%d %H:%M:%S}": r
                for r in ss.curation_runs
            }
            followup_labels = {
                label: data for label, data in ss.follow_ups.items()
            }
            all_labels = list(article_labels.keys()) + list(run_labels.keys()) + list(ss.follow_ups.keys())
            if all_labels:
                st.markdown("Download or delete existing records.")
                victim = st.selectbox("Select a record", all_labels)
                md_text = ""
                file_name = "placeholder.md"
                if victim in run_labels:
                    run = run_labels[victim]
                    md_parts = []
                    md_parts.append("### Curation Result")
                    if isinstance(run["df"], pd.DataFrame) and not run["df"].empty:
                        md_parts.append(run["df"].to_markdown(index=False))
                    md_parts.append("### Step-by-Step Reasoning")
                    md_parts.append(run["logs"])
                    md_text = "\n\n".join(md_parts)
                    md_text = prettify_md(md_text)
                    file_name = f"{run['task'].replace(' ', '_')}_{run['timestamp']:%Y%m%d_%H%M%S}.md"
                elif victim in followup_labels:
                    followup = followup_labels[victim]
                    run = followup["run"]
                    md_parts = ["### Follow-up Chat Log"]
                    for msg in followup["history"]:
                        if msg.get("role") == "system":
                            continue
                        role = msg["role"].capitalize()
                        content = msg.get("content", "")
                        md_parts.append(f"**{role}:**\n\n{content}\n")
                    md_text = "\n\n---\n\n".join(md_parts)
                    file_name = f"{run['task'].replace(' ', '_')}_{run['timestamp']:%Y%m%d_%H%M%S}_followup.md"

                st.download_button(
                    label="Download",
                    data=md_text,
                    file_name=file_name,
                    mime="text/markdown",
                    use_container_width=True,
                    disabled=not (victim in run_labels or victim in followup_labels),
                    key="download-md"
                )
                if st.button("Delete", use_container_width=True):
                    if victim in article_labels:
                        ss.retrieved_articles.pop(article_labels[victim], None)
                    elif victim in run_labels:
                        ss.curation_runs.remove(run_labels[victim])
                    elif victim in ss.follow_ups:
                        ss.follow_ups.pop(victim, None)
                    st.rerun()
            else:
                st.info("Use ‘Access Article’ first")

    # ---------------- Main Pane ----------------
    if ss.oneclick_curation_info is not None:
        st.info(ss.oneclick_curation_info)
    if ss.oneclick_curation_results:
        cur_pmid = None
        for pmid, job_name, result in ss.oneclick_curation_results:
            result: PKPECuratedTables = result
            markdown_df = result["curated_table"]
            df = markdown_to_dataframe(markdown_df)
            if df.empty:
                continue

            if cur_pmid != pmid:
                cur_pmid = pmid
                st.markdown(f"### {pmid}")

            st.markdown(f"### {job_name}")
            st.dataframe(df)
            with st.expander(f"Explanation & Suggested Fix"):
                st.markdown(f"#####  Explanation")
                st.markdown(result["explanation"])
                st.markdown("---")
                st.markdown(f"#####  Suggested Fix")
                st.markdown(result["suggested_fix"])
    
    if ss.oneclick_curation_info is None and len(ss.oneclick_curation_results) == 0:
        if  not ss.retrieved_articles:
            st.info("Use ‘Access Article’ first")
    

    for article_id, data in ss.retrieved_articles.items():
        with st.expander(f"Article Preview ({article_id})", expanded=False):
            if data["title"]:
                # st.markdown(f"### {escape_markdown(data['title'])}")
                st.markdown(
                    f"""
                    <h3>
                        {escape_markdown(data['title'])}<a href="https://pubmed.ncbi.nlm.nih.gov/{data['article_id']}" target="_blank"
                           style="font-size: 0.5em; margin-left: 5px;">View on PubMed</a>
                    </h3>
                    <br>
                    """,
                    unsafe_allow_html=True
                )
            if data["abstract"]:
                st.markdown("#####  Abstract")
                st.markdown(escape_markdown(data["abstract"]))

            if data["tables"]:
                st.markdown("---")
                for idx, tbl in enumerate(data["tables"], start=1):
                    st.markdown(f"#####  Table {idx}")
                    if tbl.get("caption"):
                        st.markdown(escape_markdown(tbl["caption"]))
                    if "table" in tbl:
                        try:
                            st.dataframe(tbl["table"])
                        except Exception:
                            st.write(tbl["table"])
                    if tbl.get("footnote"):
                        st.markdown(escape_markdown(tbl["footnote"]))
                    if idx < len(data["tables"]):
                        st.markdown("---")

                st.markdown("---")
                for idx, tbl in enumerate(data["tables"], start=1):
                    st.markdown(f"#####  Table {idx} HTML")
                    if tbl.get("raw_tag"):
                        st.text_area(
                            label="",
                            value=tbl["raw_tag"],
                            key=f"html-display-{article_id}-{idx}",
                            height=150,
                        )
                    if idx < len(data["tables"]):
                        st.markdown("---")

    # — Display curation outputs —
    if ss.curation_runs:
        for run in ss.curation_runs:
            with st.expander(f"{run['task']} ({run['article_id']}) @ {run['timestamp']:%Y-%m-%d %H:%M:%S}"):

                if run["title"]:
                    # st.markdown(f"### {escape_markdown(run['title'])}")
                    st.markdown(
                        f"""
                        <h3>
                            {escape_markdown(run['title'])}<a href="https://pubmed.ncbi.nlm.nih.gov/{run['article_id']}" target="_blank"
                               style="font-size: 0.5em; margin-left: 5px;">
                               View on PubMed</a>
                        </h3>
                        <br>
                        """,
                        unsafe_allow_html=True
                    )

                if isinstance(run['df'], pd.DataFrame) and not run['df'].empty:
                    st.markdown(f"#####  {run['task']}")
                    st.dataframe(run['df'])
                    st.markdown("---")
                st.markdown(f"#####  Step-by-Step Reasoning")
                # st.markdown(convert_log_to_markdown(run['logs']))
                convert_log_to_markdown(run['logs'])

    # — Follow-up Sessions —
    for key, fu in ss.follow_ups.items():
        with st.expander(key, expanded=True):
            # Session state for input tracking
            if "chat_input_buffer" not in fu:
                fu["chat_input_buffer"] = ""

            st.markdown("""
                <style>
                .chat-container {
                    height: 60vh;
                    overflow-y: auto;
                    padding: 1rem;
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    background-color: #f9f9f9;
                    font-family: sans-serif;
                    margin-bottom: 1rem;
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }
                
                .msg-user {
                    align-self: flex-end;
                    background-color: #e6f0ff;
                    color: #262730;
                    padding: 0.6em 0.9em;
                    border-radius: 1rem;
                    border-bottom-right-radius: 0;
                    max-width: 70%;
                    word-wrap: break-word;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                }
                
                .msg-assistant {
                    align-self: flex-start;
                    background-color: #f1f3f6;
                    color: #262730;
                    padding: 0.6em 0.9em;
                    border-radius: 1rem;
                    border-bottom-left-radius: 0;
                    max-width: 70%;
                    word-wrap: break-word;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                }
                </style>
            """, unsafe_allow_html=True)

            full_history = fu["history"][:]
            if "pending_user_msg" in fu:
                full_history.append({"role": "user", "content": fu["pending_user_msg"]})

            chat_html = '<div class="chat-container">'
            for msg in full_history:
                if msg["role"] == "user":
                    chat_html += f'<div class="msg-user">{msg["content"]}</div>'
                elif msg["role"] == "assistant":
                    chat_html += f'<div class="msg-assistant">{msg["content"]}</div>'
            chat_html += '</div>'

            st.markdown(chat_html, unsafe_allow_html=True)

            input_col1, input_col2 = st.columns([11, 1])
            with input_col1:
                fu["chat_input_buffer"] = st.text_input("Your message",
                                                        value=fu["chat_input_buffer"],
                                                        label_visibility="collapsed",
                                                        placeholder="Ask anything",
                                                        key=f"chat_input_buffer_{key}")
            with input_col2:
                send_clicked = st.button("Send", use_container_width=True, key=f"send_button_{key}")

            if send_clicked and fu["chat_input_buffer"].strip():
                user_msg = fu["chat_input_buffer"].strip()
                fu["pending_user_msg"] = user_msg
                fu["chat_input_buffer"] = ""
                st.rerun()

            if "pending_user_msg" in fu:
                msg = fu.pop("pending_user_msg")
                fu["history"].append({"role": "user", "content": msg})

                client_4o, model_4o, *_ = get_client_and_model()
                langchain_history = []
                for m in fu["history"]:
                    if m["role"] == "user":
                        langchain_history.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        langchain_history.append(AIMessage(content=m["content"]))
                    elif m["role"] == "system":
                        langchain_history.append(SystemMessage(content=m["content"]))

                assistant_reply = client_4o.invoke(langchain_history).content
                fu["history"].append({"role": "assistant", "content": assistant_reply})
                st.rerun()


if __name__ == "__main__":
    main_tab()
