import pytest

from extractor.agents.pk_pe_agents.pk_pe_design_step import PKPEDesignStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState, PaperTypeEnum
from extractor.utils import convert_sections_to_full_text

# @pytest.mark.skip()
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


