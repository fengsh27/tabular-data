
import pytest
from pathlib import Path

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PaperTypeEnum
from extractor.constants import PipelineTypeEnum
from extractor.database.pmid_db import PMIDDB
from extractor.agents_manager.pk_pe_manager import PKPEManager
from extractor.request_gpt_oss import get_gpt_oss

@pytest.mark.skip()
def test_pk_pe_manager_run_pk_workflows(llm, pmid_db):
    pk_pe_manager = PKPEManager(llm, pmid_db)
    res = pk_pe_manager._run_pk_workflows("17635501")
    assert res

@pytest.mark.skip()
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
    
@pytest.mark.skip()
def test_pk_pe_manager_identification_step(llm, llm_agent, pmid_db):
    pmid = "39843580"
    pk_pe_manager = PKPEManager(
        pipeline_llm=llm,
        agent_llm=llm_agent,
        pmid_db=pmid_db,
    )
    pk_pe_manager._extract_pmid_info(pmid)
    state = pk_pe_manager._identification_and_design_step(pmid)
    assert "paper_type" in state and state["paper_type"] != None

@pytest.mark.parametrize("pmid", [
    "11849190",
    "10971311",
    "18426260",
    "23200982",
    "24989434",
    "32153014",
    "34746508",
    "33253437",
    "32056930",
])
def test_pk_pe_manager_run_pk_ind_workflow(llm, llm_agent, pmid_db, pmid):
    with open(f"./system_tests/data/{pmid}.html", "r") as fobj:
        html_content = fobj.read()
    
    pk_pe_manager = PKPEManager(
        pipeline_llm=llm,
        agent_llm=llm_agent,
        pmid_db=pmid_db,
    )
    res = pk_pe_manager.run(
        pmid, 
        html_content=html_content,
        pipeline_types=[PipelineTypeEnum.PK_INDIVIDUAL]
    )
    assert res
    assert res[PipelineTypeEnum.PK_INDIVIDUAL]["curated_table"] is not None
    df = markdown_to_dataframe(res[PipelineTypeEnum.PK_INDIVIDUAL]["curated_table"])
    csv_path = Path(__file__).parent / "data" / "2026-1-1" / f"{pmid}_qwen3.csv"
    df.to_csv(csv_path, index=False)
