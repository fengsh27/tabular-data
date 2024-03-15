import pytest
from src.doi_article_retriever import DOIRetriever

@pytest.mark.skip("EPUB file format, even puppeteer can't get html version")
def test_DOIRetriever():
    with open("tests/35880962.html", "r") as fobj:
        html = fobj.read()
        retriever = DOIRetriever(html)
        assert retriever.is_applicable()
        res, text, code = retriever.extract()
        assert res



