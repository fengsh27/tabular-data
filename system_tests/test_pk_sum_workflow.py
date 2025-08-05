import pytest
import logging

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow

logger = logging.getLogger(__name__)


@pytest.mark.skip()
def test_PKSumWorkflow(llm, html_content, caption, step_callback):
    workflow = PKSumWorkflow(llm=llm)
    workflow.build()

    df = workflow.go(html_content, caption, step_callback)
    print(df)
    logger.info("\n\n" + dataframe_to_markdown(df))


@pytest.mark.skip()
def test_PKSumWorkflow1(llm, html_content1, caption1, step_callback):
    workflow = PKSumWorkflow(llm=llm)
    workflow.build()

    df = workflow.go(html_content1, caption1, step_callback)
    print(df)
    logger.info("\n\n" + dataframe_to_markdown(df))


@pytest.mark.skip()
def test_PKSumWorkflow_29943508(
    llm, html_content_29943508, caption_29943508, step_callback
):
    workflow = PKSumWorkflow(llm=llm)
    workflow.build()

    df = workflow.go(html_content_29943508, caption_29943508, step_callback)
    print(df)
    logger.info("\n\n" + dataframe_to_markdown(df))


@pytest.mark.skip()
def test_PKSumWorkflow_16143486_table_4(
    llm, html_content_16143486_table_4, caption_16143486_table_4, step_callback
):
    workflow = PKSumWorkflow(llm=llm)
    workflow.build()
    df = workflow.go(
        html_content_16143486_table_4, caption_16143486_table_4, step_callback
    )
    print(df)
    logger.info("\n\n" + dataframe_to_markdown(df))


@pytest.mark.skip()
def test_PKSumWorkflow_30825333_table_2(
    llm,
    html_content_30825333_table_2,
    caption_30825333_table_2,
    step_callback,
):
    workflow = PKSumWorkflow(llm)
    workflow.build()
    df = workflow.go(
        html_content_30825333_table_2,
        caption_30825333_table_2,
        step_callback,
    )
    print(df)


def test_PKSumWorkflow_22050870_table_3(
    llm,
    html_content_22050870_table_3,
    caption_22050870_table_3,
    title_22050870,
    step_callback,
):
    workflow = PKSumWorkflow(llm)
    workflow.build()
    df = workflow.go(
        html_content=html_content_22050870_table_3,
        caption_and_footnote=caption_22050870_table_3,
        title=title_22050870,
        step_callback=step_callback,
    )
    print(df)
