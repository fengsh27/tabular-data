
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow

def test_PKSumWorkflow(llm, html_content, caption):
    workflow = PKSumWorkflow(llm=llm)
    workflow.build()

    df = workflow.go(html_content, caption)

