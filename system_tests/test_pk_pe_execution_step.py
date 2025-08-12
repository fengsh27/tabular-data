
from pathlib import Path
import pytest

from unittest import TestCase

from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState
from extractor.agents.pk_pe_agents.pk_pe_execution_step import PKPEExecutionStep
from extractor.agents.pk_pe_agents.pk_pe_agent_tools import (
    PKSummaryTablesCurationTool,
)
from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.utils import convert_html_to_text_no_table, remove_references

@pytest.fixture
def pmid_db():
    return PMIDDB(db_path=Path("./system_tests/data/pmid_info.db"))

@pytest.fixture
def paper_17635501(pmid_db):
    pmid_db: PMIDDB = pmid_db
    pmid = "17635501"
    pmid_info = pmid_db.select_pmid_info(pmid)
    if pmid_info is not None:
        return pmid_info
    
    retriever = ArticleRetriever()
    res, html_content, code = retriever.request_article("17635501")
    if not res:
        raise ValueError("Failed to retrieve paper 17635501")
    extractor = HtmlTableExtractor()
    tables = extractor.extract_tables(html_content)
    sections = extractor.extract_sections(html_content)
    abstract = extractor.extract_abstract(html_content)
    title = extractor.extract_title(html_content)
    full_text = convert_html_to_text_no_table(html_content)
    full_text = remove_references(full_text)
    pmid_db.insert_pmid_info(pmid, title, abstract, full_text, tables, sections)
    return pmid, title, abstract, full_text, tables, sections

def test_pk_pe_execution_step(
    llm,
    step_callback,
    pmid_db,
    paper_17635501,
):
    step = PKPEExecutionStep(
        llm=llm,
        tool=PKSummaryTablesCurationTool(
            llm=llm,
            output_callback=step_callback,
            pmid="17635501",
            pmid_db=pmid_db,
        ),
    )

    state = PKPECurationWorkflowState(
        pmid="17635501",
        pk_pe_type="PK",
        paper_title=paper_17635501[1],
        paper_abstract=paper_17635501[2],
        source_tables=paper_17635501[4],
        step_output_callback=step_callback,
    )

    state = step.execute(state)


