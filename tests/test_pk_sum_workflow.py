import logging
from typing import Optional

from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow

logger = logging.getLogger(__name__)

def print_step(
    step_name: Optional[str]=None, 
    step_description: Optional[str]=None,
    step_output: Optional[str]=None,
    step_reasoning_process: Optional[str]=None,
    token_usage: Optional[dict]=None,
):
    if step_name is not None:
        logger.info("=" * 64)
        logger.info(step_name)
    if step_description is not None:
        logger.info(step_description)
    if token_usage is not None:
        logger.info(f"total tokens: {token_usage['total_tokens']}")
    if step_reasoning_process is not None:
        logger.info(f"\n\n{step_reasoning_process}\n\n")
    if step_output is not None:
        logger.info(step_output)

def test_PKSumWorkflow(llm, html_content, caption):
    workflow = PKSumWorkflow(llm=llm)
    workflow.build()

    df = workflow.go(html_content, caption, print_step)

