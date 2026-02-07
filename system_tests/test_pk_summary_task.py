
import pytest
from pathlib import Path

from extractor.agents_manager.pk_summary_task import PKSummaryTask
from extractor.agents.agent_utils import extract_pmid_info_to_db
from TabFuncFlow.utils.table_utils import markdown_to_dataframe

@pytest.mark.parametrize("pmid", [
    # "16143486",
    # "17635501",
    # "22050870",
    # "29943508",
    # "30825333",
    # "34114632",
    # "34183327",
    # "35465728",
    # "35489632",

    # "31935538",
    # "29718415",
    # "31112621",
    "36181377",
])
def test_PKSummaryTask(
    llm,
    step_callback,
    pmid_db,
    pmid,
):
    paper_id, title, abstract, full_text, tables, sections = extract_pmid_info_to_db(pmid, pmid_db)
    assert paper_id is not None

    task = PKSummaryTask(llm=llm, output_callback=step_callback, pmid_db=pmid_db)
    res, curated_table, explanation, suggested_fix = task.run(pmid)
    if curated_table is not None:
        df = markdown_to_dataframe(curated_table)
        csv_path = Path(__file__).parent / "data" / f"{pmid}_gpt4o.csv"
        df.to_csv(csv_path, index=False)
        if not res:
            output_path = Path(__file__).parent / "data" / f"{pmid}_gpt4o_output.log"
            output_path.write_text(f"explanation: {explanation}\n\nsuggested_fix: {suggested_fix}")
    assert res
    assert curated_table is not None

