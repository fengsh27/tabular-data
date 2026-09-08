from pathlib import Path
import pytest
from app_script_pmids import prepare_data_by_pmids_csv_file
from extractor.database.pmid_db import PMIDDB

@pytest.mark.skip()
def test_prepare_data_by_pmids_csv_file_inserts_and_skips(tmp_path):
    csv_path = Path(__file__).parent / "data" / "pmids_html_fixture.csv"
    db_path = tmp_path / "pmid_info.db"
    pmid_db = PMIDDB(db_path)

    prepare_data_by_pmids_csv_file(str(csv_path), pmid_db)

    info = pmid_db.select_pmid_info("17158945")
    assert info is not None
    pmid, title, abstract, full_text, tables, sections = info

    assert pmid == "17158945"
    assert title is not None and title.strip()
    assert abstract is not None and abstract.strip()
    assert full_text is not None and full_text.strip()
    assert isinstance(tables, list)
    assert isinstance(sections, list)

    # Existing PMID should be skipped even if HTML path is invalid.
    csv_bad = tmp_path / "pmids_html_fixture_bad.csv"
    csv_bad.write_text("pmid,html_file\n17158945,missing.html\n", encoding="utf-8")
    prepare_data_by_pmids_csv_file(str(csv_bad), pmid_db)
