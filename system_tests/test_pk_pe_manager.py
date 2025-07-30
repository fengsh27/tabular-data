
import pytest

from extractor.agents_manager.pk_pe_manager import PKPEManager

def test_pk_pe_manager(llm):
    pk_pe_manager = PKPEManager(llm)
    res = pk_pe_manager._extract_pmid_info("22050870")
    assert res