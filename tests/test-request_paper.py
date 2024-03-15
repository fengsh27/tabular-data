
import pytest

from src.request_paper import PaperRetriver
from src.article_stamper import Stamper

def test_PaperRetriever_for_Elsevier():
    id = "25448584" 
    stamper = Stamper(id)
    retriever = PaperRetriver(stamper)

    res, text, code =retriever.request_paper(id)
    assert res
