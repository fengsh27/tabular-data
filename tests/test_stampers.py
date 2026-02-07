import pytest
import os
from os import path
import shutil
from extractor.stampers import ArticleStamper

TESTS_TEMP = "./tests/tmp"


@pytest.fixture(scope="module")
def tmp_folder():
    if not path.exists(TESTS_TEMP):
        os.mkdir(TESTS_TEMP)
    yield
    if path.exists(TESTS_TEMP):
        shutil.rmtree(TESTS_TEMP)


def test_stamper(tmp_folder):
    stamper = ArticleStamper(TESTS_TEMP, True)
    stamper.pmid = "123456"
    stamper.output_html("<html></html")
    stamper.output_prompts([{"role": "user", "content": "Hi"}])
    stamper.output_result("empty result")
    the_folder = path.join(TESTS_TEMP, "123456")
    assert path.exists(the_folder)
    res = list(os.walk(the_folder))[0]
    files = res[2]
    assert len(files) == 3
