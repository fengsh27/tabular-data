from unittest.mock import patch
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.prompts_utils import _generate_table_prompts
from extractor.utils import convert_html_to_text


def get_html_content(id: str):
    with open(f"./tests/data/{id}.html", "r") as fobj:
        return fobj.read()


class MockResponse:
    def __init__(self, text, status_code):
        self.text = text
        self.status_code = status_code


@patch("extractor.pmid_extractor.article_retriever.make_article_request")
@patch("extractor.pmid_extractor.article_retriever.make_get_request")
def test_get_html_content(mock_make_get_request, mock_make_article_request):
    id = "17158945"
    mock_make_get_request.return_value = MockResponse(get_html_content(id), 200)

    retriever = ArticleRetriever()
    res, html_content, code = retriever.request_article(id)
    text = convert_html_to_text(html_content)
    extractor = HtmlTableExtractor()
    tables = extractor.extract_tables(html_content)
    prompts = "Here are tables in the paper (including their caption and footnote)\n"
    for table in tables:
        prompts += _generate_table_prompts(table)
    assert res
    assert html_content is not None
    assert text is not None
    assert len(tables) == 2


@patch("extractor.pmid_extractor.article_retriever.make_article_request")
@patch("extractor.pmid_extractor.article_retriever.make_get_request")
def test_get_html_content_1(mock_make_get_request, mock_make_article_request):
    id = "35880962"
    mock_make_get_request.return_value = MockResponse(get_html_content(id), 200)
    retriever = ArticleRetriever()
    res, html_content, code = retriever.request_article(id)
    text = convert_html_to_text(html_content)
    extractor = HtmlTableExtractor()
    tables = extractor.extract_tables(html_content)
    prompts = "Here are tables in the paper (including their caption and footnote)\n"
    for table in tables:
        prompts += _generate_table_prompts(table)
    assert res
    assert html_content is not None
    assert text is not None
    assert len(tables) == 0
