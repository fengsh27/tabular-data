import pytest


from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import extract_pmid_info_to_db
from extractor.agents.pk_pe_agents.pk_pe_correction_step import PKPECuratedTablesCorrectionStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState

REASONING_PROCESS = """
"""

def test_pk_pe_correction_step_on_29718415(
    llm, 
    step_callback,
    pmid_db,
    md_table_curated_29718415_table_3,
    verification_reasoning_process_29718415_table_3,
):
    pmid = "29718415"
    pmid, title, abstract, full_text, tables, sections = extract_pmid_info_to_db(pmid, pmid_db)
    assert pmid is not None
    print_step = step_callback

    # prepare tables
    source_tables = []
    for ix in [2]:
        table = tables[ix]
        caption = "\n".join([table["caption"], table["footnote"]])
        source_table = dataframe_to_markdown(table["table"])
        source_tables.append(f"caption: \n{caption}\n\n table: \n{source_table}")

    step = PKPECuratedTablesCorrectionStep(
        llm=llm,
        pmid=pmid,
        domain="pharmacokinetics",
    )

    state: PKPECurationWorkflowState = {
        "paper_title": title,
        "paper_abstract": abstract,
        "source_tables": source_tables,
        "curated_table": md_table_curated_29718415_table_3,
        "verification_reasoning_process": verification_reasoning_process_29718415_table_3,
        "step_output_callback": print_step,
    }

    state = step.execute(state)

    assert state["curated_table"] is not None
    
    