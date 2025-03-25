import logging
from typing import Optional

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow

logger = logging.getLogger(__name__)

def test_PKSumWorkflow(llm, html_content, caption, step_callback):
    workflow = PKSumWorkflow(llm=llm)
    workflow.build()

    df = workflow.go(html_content, caption, step_callback)
    print(df)
    logger.info("\n\n" + dataframe_to_markdown(df))

