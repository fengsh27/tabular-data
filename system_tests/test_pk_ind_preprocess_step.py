import pytest

from extractor.agents.pk_individual.pk_ind_preprocess_step import (
    PKIndPreprocessStep,
    PKIndWorkflowState,
)

def test_PKIndPreprocessStep_18426260_table_1(
    llm,
    caption_18426260_table_1,
    md_table_18426260_table_1,
    full_text_excluding_tables_18426260,
    step_callback,
):
    step = PKIndPreprocessStep()
    state = PKIndWorkflowState(
        llm=llm,
        caption=caption_18426260_table_1,
        md_table=md_table_18426260_table_1,
        full_text=full_text_excluding_tables_18426260,
        step_callback=step_callback,
    )

    state = step.execute(state)

    assert(state is not None)
    assert(state["md_table"] is not None)

