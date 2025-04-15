
import pytest

from extractor.html_table_extractor import HtmlTableExtractor
from extractor.table_utils import select_pk_summary_tables

def test_select_pk_summary_tables(
    llm,
):
    with open("tests/data/17635501.html", "r") as fobj:
        html_content = fobj.read()

    tables_extractor = HtmlTableExtractor()
    tables = tables_extractor.extract_tables(html_content)
    tables, table_indices, usage = select_pk_summary_tables(tables, llm)

    assert len(table_indices) == 1
    assert int(table_indices[0]) == 2



