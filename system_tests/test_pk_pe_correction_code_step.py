import pytest


from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import extract_pmid_info_to_db
from extractor.agents.pk_pe_agents.pk_pe_correction_code_step import PKPECuratedTablesCorrectionCodeStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState

REASONING_PROCESS = """
"""

def test_pk_pe_correction_step_on_29718415(
    llm,
    step_callback,
    pmid_db,
    title_29100749,
    abstract_29100749,
    source_table_29100749_table_2,
    curated_table_29100749,
    verification_reasoning_29100749,
):
    pmid = "29100749"
    
    step = PKPECuratedTablesCorrectionCodeStep(
        llm=llm,
        pmid=pmid,
        domain="pharmacokinetics",
    )

    state: PKPECurationWorkflowState = {
        "paper_title": title_29100749,
        "paper_abstract": abstract_29100749,
        "source_tables": source_table_29100749_table_2,
        "curated_table": curated_table_29100749,
        "verification_reasoning_process": verification_reasoning_29100749,
        "step_output_callback": step_callback,
    }

    state = step.execute(state)

    assert state["curated_table"] is not None
    
    