
import pytest

from extractor.agents.pk_pe_agents.pk_pe_agent_tools import (
    FullTextCurationTool,
    PKSummaryTablesCurationTool,
    PKPopulationSummaryCurationTool,
)
from extractor.agents.pk_specimen_summary.pk_spec_sum_workflow import PKSpecSumWorkflow

# @pytest.mark.skip()
def test_pk_summary_tables_curation_tool(llm, step_callback):
    tool = PKSummaryTablesCurationTool(pmid="22050870", llm=llm, output_callback=step_callback)
    res = tool.run()
    assert res is not None

@pytest.mark.skip()
def test_pk_population_summary_workflow_tool(llm, step_callback):
    tool = PKPopulationSummaryCurationTool(pmid="22050870", llm=llm, output_callback=step_callback)
    res = tool.run()
    assert res is not None

@pytest.mark.skip()
def test_full_text_workflow_tool(llm, step_callback):
    tool = FullTextCurationTool(
        pmid="22050870",
        cls=PKSpecSumWorkflow,
        tool_name="PK Specimen Summary Curation Tool",
        tool_description="This tool is used to extract the specimen summary data from the source paper.",
        llm=llm,
        output_callback=step_callback,
    )
    res = tool.run()
    assert res is not None


