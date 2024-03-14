
import pytest

from src.request_paper import (
    request_paper,
    convert_html_to_text,
    extract_tables_from_html
)
from src.prompts_utils import (
    _generate_table_prompts
)

@pytest.mark.skip("temporary skipped")
def test_get_html_content():
    id = "17158945"
    res, html_content, code = request_paper(id)
    text = convert_html_to_text(html_content, True)
    tables = extract_tables_from_html(html_content)
    prompts = "Here are tables in the paper (including their caption and footnote)\n"
    for table in tables:
        prompts += _generate_table_prompts(table)
    assert res
    assert html_content is not None
    assert text is not None


