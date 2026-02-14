import pytest


from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import extract_pmid_info_to_db
from extractor.agents.pk_pe_agents.pk_pe_correction_code_step import PKPECuratedTablesCorrectionCodeStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState

REASONING_PROCESS = """
"""
@pytest.mark.skip()
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
    
@pytest.mark.skip()
def test_pk_pe_correction_code_step_34746508(
    llm,
    step_callback,
    title_34746508,
    abstract_34746508,
    md_table_34746508_table_1,
    md_table_34746508_table_2,
    caption_34746508_table_1,
    caption_34746508_table_2,
    pk_individual_curated_table_34746508,
    pk_individual_verification_reasoning_34746508,
):
    pmid = "34746508"
    
    step = PKPECuratedTablesCorrectionCodeStep(
        llm=llm,
        pmid=pmid,
        domain="pharmacokinetics",
    )

    state: PKPECurationWorkflowState = {
        "paper_title": title_34746508,
        "paper_abstract": abstract_34746508,
        "source_tables": [md_table_34746508_table_1, md_table_34746508_table_2],
        "curated_table": pk_individual_curated_table_34746508,
        "verification_reasoning_process": pk_individual_verification_reasoning_34746508,
        "step_output_callback": step_callback,
    }

    state = step.execute(state)

    assert state["curated_table"] is not None
    
    assert state["verification_reasoning_process"] is not None


def test_pk_individual_correction_code_step_23200982(
    llm,
    step_callback,
    title_23200982,
    abstract_23200982,
    # caption_23200982_table_1,
    # caption_23200982_table_2,
    md_table_23200982_table_3,
    md_table_23200982_table_1,
    md_table_23200982_table_2,
    md_curated_table_pk_individual_23200982,
    verification_explanation_23200982,
    verification_final_answer_23200982,
    verification_suggested_fix_23200982,
):
    pmid = "23200982"
    
    step = PKPECuratedTablesCorrectionCodeStep(
        llm=llm,
        pmid=pmid,
        domain="pharmacokinetics",
    )

    state: PKPECurationWorkflowState = {
        "paper_title": title_23200982,
        "paper_abstract": abstract_23200982,
        "source_tables": [md_table_23200982_table_1, md_table_23200982_table_2, md_table_23200982_table_3],
        "curated_table": md_curated_table_pk_individual_23200982,
        "verification_reasoning_process": verification_explanation_23200982,
        "final_answer": verification_final_answer_23200982,
        "suggested_fix": verification_suggested_fix_23200982,
        "step_output_callback": step_callback,
    }

    state = step.execute(state)

    assert state["curated_table"] is not None
    
    assert state["verification_reasoning_process"] is not None



