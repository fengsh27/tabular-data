import pytest

from extractor.agents.pk_individual.pk_ind_workflow import PKIndWorkflow
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

# @pytest.mark.skip()
def test_PKIndWorkflow_29100749_table_2(
    llm, 
    title_29100749,
    caption_29100749_table_2, 
    source_table_29100749_table_2,
    step_callback
):
    workflow = PKIndWorkflow(llm=llm)
    workflow.build()

    df = workflow.go_md_table(
        md_table=source_table_29100749_table_2,
        caption_and_footnote=caption_29100749_table_2,
        title=title_29100749,
        step_callback=step_callback,
    )
    print(df)
    assert df is not None
    assert df.empty is False

@pytest.mark.skip()
def test_PKIndWorkflow_32635742_table_0(
    llm,
    title_32635742,
    md_table_32635742_table_0,
    caption_32635742_table_0,
    step_callback,
    full_text_32635742,
):
    workflow = PKIndWorkflow(llm=llm)
    workflow.build()
    df = workflow.go_md_table(
        md_table=md_table_32635742_table_0,
        caption_and_footnote=caption_32635742_table_0,
        title=title_32635742,
        step_callback=step_callback,
        full_text=full_text_32635742,
    )
    print(df)
    assert df is not None
    assert df.empty is False