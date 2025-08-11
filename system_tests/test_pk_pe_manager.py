
import pytest
from pathlib import Path

from extractor.database.pmid_db import PMIDDB
from extractor.agents_manager.pk_pe_manager import PKPEManager

def test_pk_pe_manager_run_pk_workflows(llm, pmid_db):
    pk_pe_manager = PKPEManager(llm, pmid_db)
    res = pk_pe_manager._run_pk_workflows("17635501")
    assert res