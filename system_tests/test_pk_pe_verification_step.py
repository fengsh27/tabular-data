
import pytest
import logging

from extractor.agents.pk_pe_agents.pk_pe_verification_step import PKPECuratedTablesVerificationStep
from extractor.agents.pk_pe_agents.pk_pe_correction_code_step import PKPECuratedTablesCorrectionCodeStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import FinalAnswerEnum, PKPECurationWorkflowState
from extractor.request_gpt_oss import get_gpt_oss, get_gpt_qwen_30b

logger = logging.getLogger(__name__)

@pytest.mark.skip()
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
        "llm": llm,
        "paper_title": paper_title_17635501,
        "paper_abstract": paper_abstract_17635501,
        "source_tables": md_table_17635501_table_3,
        "curated_table": curated_table_17635501_table_3,
        "step_output_callback": step_callback,
    }
    state = step.execute(state)

    assert state["curated_table"] is not None

# @pytest.mark.skip()
def test_pk_pe_verification_step_on_29100749(
    llm,
    step_callback,
    pmid_db,
    title_29100749,
    abstract_29100749,
    source_table_29100749_table_2,
    curated_table_29100749,
):
    verification_step = PKPECuratedTablesVerificationStep(
        llm=llm,
        pmid="29100749",
        domain="pharmacokinetic population and individual",
    )
    correction_step = PKPECuratedTablesCorrectionCodeStep(
        llm=llm,
        pmid="29100749",
        domain="pharmacokinetic population and individual",
    )
    state: PKPECurationWorkflowState = {
        "llm": llm,
        "paper_title": title_29100749,
        "paper_abstract": abstract_29100749,
        "source_tables": source_table_29100749_table_2,
        "curated_table": curated_table_29100749,
        "step_output_callback": step_callback,
    }
    state = verification_step.execute(state)
    n = 0
    while n < 5 and not ("final_answer" in state and \
        state["final_answer"] in [FinalAnswerEnum.Correct, FinalAnswerEnum.Error]):
        state = correction_step.execute(state)
        state = verification_step.execute(state)
        n += 1
    assert state["curated_table"] is not None


