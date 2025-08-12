
import pytest

from extractor.agents.pk_pe_agents.pk_pe_verification_step import PKPECuratedTablesVerificationStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState

def test_pk_pe_verification_step(
    llm, 
    step_callback,
    paper_title_17635501,
    paper_abstract_17635501,
    md_table_17635501_table_3,
    curated_table_17635501_table_3,
):
    step = PKPECuratedTablesVerificationStep(
        llm=llm,
        pmid="17635501",
        domain="pharmacokinetics",
    )

    state: PKPECurationWorkflowState = {
        "paper_title": paper_title_17635501,
        "paper_abstract": paper_abstract_17635501,
        "source_tables": md_table_17635501_table_3,
        "curated_table": curated_table_17635501_table_3,
        "step_output_callback": step_callback,
    }
    state = step.execute(state)

    assert state["curated_table"] is not None
    
