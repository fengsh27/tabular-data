import pytest
from pathlib import Path

from extractor.database.curation_db import CurationDB

@pytest.fixture
def curation_db():
    db_path = Path("./tests/data/curation_data.db")
    db_path.touch()
    yield CurationDB(db_path=Path("./tests/data/curation_data.db"))
    db_path.unlink()


def test_curation_db(curation_db):
    assert curation_db is not None
    assert curation_db.get_curation_count() == 0
    assert curation_db.insert_curation_data("1234567890", "test", "test", "test", "test", "test")
    assert curation_db.get_curation_count() == 1