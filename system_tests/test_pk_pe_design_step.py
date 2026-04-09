import pytest
import logging
from pathlib import Path

from extractor.agents.pk_pe_agents.pk_pe_design_step import PKPEDesignStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState, PaperTypeEnum
from extractor.utils import convert_sections_to_full_text
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.pmid_extractor.table_utils import format_source_tables

logger = logging.getLogger(__name__)

@pytest.mark.skip()
def test_pk_pe_design_step_on_10971311(
    llm_agent,
    step_callback,
    title_10971311,
    abstract_10971311,
    sections_10971311,
):
    step = PKPEDesignStep(llm=llm_agent)
    full_text = convert_sections_to_full_text(sections_10971311)
    state = PKPECurationWorkflowState(
        pmid="10971311",
        paper_title=title_10971311,
        paper_abstract=abstract_10971311,
        full_text=full_text,
        step_output_callback=step_callback,
        paper_type=PaperTypeEnum.PK,
    )
    state = step.execute(state)
    assert state["pipeline_tools"] is not None
    assert len(state["pipeline_tools"]) > 0
    assert all(tool in state["pipeline_tools"] for tool in ["pk_specimen_summary", "pk_specimen_individual"])

@pytest.mark.skip()
def test_pk_pe_design_step_on_18426260(
    llm_agent,
    step_callback,
    title_18426260,
    abstract_18426260,
    sections_18426260,
):
    step = PKPEDesignStep(llm=llm_agent)
    full_text = convert_sections_to_full_text(sections_18426260)
    state = PKPECurationWorkflowState(
        pmid="18426260",
        paper_title=title_18426260,
        paper_abstract=abstract_18426260,
        full_text=full_text,
        step_output_callback=step_callback,
        paper_type=PaperTypeEnum.PK,
    )
    state = step.execute(state)
    assert state["pipeline_tools"] is not None
    assert len(state["pipeline_tools"]) > 0
    # assert all(tool in state["pipeline_tools"] for tool in ["pk_individual", "pk_specimen_summary", "pk_specimen_individual"])

@pytest.mark.parametrize("pmid", [
    "10971311",
    "11849190",
    "18426260",
    "11849190",
    "23200982",
    "24989434",
    "32056930",
    "32153014",
    "32635742",
    "34746508",
])
def test_pk_pe_design_step_on_pmids(llm_agent, step_callback, pmid):
    html_path = Path(f"./system_tests/data/{pmid}.html")
    html_content = html_path.read_text(encoding="utf-8", errors="ignore")
    extractor = HtmlTableExtractor()
    sections = extractor.extract_sections(html_content)
    full_text = convert_sections_to_full_text(sections)
    tables = extractor.extract_tables(html_content)
    abstract = extractor.extract_abstract(html_content)
    title = extractor.extract_title(html_content)
    formatted_tables = format_source_tables(tables)
    
    state = PKPECurationWorkflowState(
        pmid=pmid,
        paper_title=title,
        paper_abstract=abstract,
        full_text=full_text,
        source_tables=formatted_tables,
        paper_type=PaperTypeEnum.PK,
        step_output_callback=step_callback,
    )
    step = PKPEDesignStep(llm=llm_agent)
    state = step.execute(state)
    assert state["pipeline_tools"] is not None
    assert len(state["pipeline_tools"]) > 0

    logger.info(f"PMID: {pmid}, Pipeline Tools: {state['pipeline_tools']}")


    
    
    
