
import pytest

from src.article_stamper import Stamper
from src.request_paper import PaperRetriver
from src.prompts_utils import (
    _generate_table_prompts
)



@pytest.mark.skip("temporary skipped")
def test_get_html_content():
    id = "17158945"
    stamper = Stamper(id)
    retriever = PaperRetriver(stamper)
    res, html_content, code = retriever.request_paper(id)
    text = retriever.convert_html_to_text(html_content, True)
    tables = retriever.extract_tables_from_html(html_content)
    prompts = "Here are tables in the paper (including their caption and footnote)\n"
    for table in tables:
        prompts += _generate_table_prompts(table)
    assert res
    assert html_content is not None
    assert text is not None

def test_get_html_content_1():
    id = "35880962"
    stamper = Stamper(id)
    retriever = PaperRetriver(stamper)
    res, html_content, code = retriever.request_paper(id)
    text = retriever.convert_html_to_text(html_content, True)
    tables = retriever.extract_tables_from_html(html_content)
    prompts = "Here are tables in the paper (including their caption and footnote)\n"
    for table in tables:
        prompts += _generate_table_prompts(table)
    assert res
    assert html_content is not None
    assert text is not None


