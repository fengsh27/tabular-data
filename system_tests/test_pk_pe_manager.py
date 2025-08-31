
import pytest
from pathlib import Path

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.constants import PipelineTypeEnum
from extractor.database.pmid_db import PMIDDB
from extractor.agents_manager.pk_pe_manager import PKPEManager

@pytest.mark.skip()
def test_pk_pe_manager_run_pk_workflows(llm, pmid_db):
    pk_pe_manager = PKPEManager(llm, pmid_db)
    res = pk_pe_manager._run_pk_workflows("17635501")
    assert res

@pytest.mark.parametrize("pmid", [
    # "22050870",
    # "29943508",
    # "30950674",
    #"34114632",
    # "34183327",
    "35465728",
    "35489632",
    "30825333",
    "16143486",
    # "17635501",
])
def test_pk_pe_manager_run_pk_workflows_1(llm, pmid_db, pmid):
    pk_pe_manager = PKPEManager(llm, pmid_db)
    res = pk_pe_manager.run(pmid, pipeline_types=[PipelineTypeEnum.PK_SUMMARY])
    assert res
    assert res[PipelineTypeEnum.PK_SUMMARY]["curated_table"] is not None
    df = markdown_to_dataframe(res[PipelineTypeEnum.PK_SUMMARY]["curated_table"])
    csv_path = Path(__file__).parent / "data" / f"{pmid}_gpt5.csv"
    df.to_csv(csv_path, index=False)
    # write explanation and suggested fix
    with open(csv_path.with_suffix(".txt"), "w") as f:
        f.write(f"correct: {res[PipelineTypeEnum.PK_SUMMARY]['correct']}\n")
        f.write(f"explanation: {res[PipelineTypeEnum.PK_SUMMARY]['explanation']}\n")
        f.write(f"suggested_fix: {res[PipelineTypeEnum.PK_SUMMARY]['suggested_fix']}")
    