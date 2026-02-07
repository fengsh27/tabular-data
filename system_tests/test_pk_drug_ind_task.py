
import pytest
from extractor.agents_manager.pk_fulltext_tool_task import PKDrugIndividualTask
from extractor.agents.agent_utils import extract_pmid_info_to_db


def test_PKDrugIndividualTask(
    llm,
    step_callback,
    pmid_db,
):
    pmid = "33710978"
    pmid, title, abstract, full_text, tables, sections = extract_pmid_info_to_db(pmid, pmid_db)
    assert pmid is not None

    task = PKDrugIndividualTask(llm=llm, output_callback=step_callback, pmid_db=pmid_db)
    res, curated_table, explanation, suggested_fix = task.run(pmid)
    assert res

