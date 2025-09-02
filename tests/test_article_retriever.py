
import pytest

from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.utils import convert_html_to_text_no_table, remove_references

@pytest.mark.parametrize("pmid", [
    "30950674", 
    "30983533",
])
def test_get_epub_data(pmid):
    retriever = ArticleRetriever()
    res, html_content, code = retriever.request_article(pmid)
    extractor = HtmlTableExtractor()
    tables = extractor.extract_tables(html_content)
    sections = extractor.extract_sections(html_content)
    abstract = extractor.extract_abstract(html_content)
    title = extractor.extract_title(html_content)
    full_text = convert_html_to_text_no_table(html_content)
    full_text = remove_references(full_text)
    assert res

