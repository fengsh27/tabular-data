
import json
from extractor.utils import concate_llm_contents

def test_concate_llm_contents_not_trucated():
    contents = [
        '[{"aaa": "...", "bbb": "...", "ccc": "..."}, {"aaa": "...", "bbb": ',
        '"...", "ccc": "..."}]',
    ]
    all_contents, all_usage, truncated = concate_llm_contents(contents=contents, usages=[24, 12])
    objs = json.loads(all_contents)
    assert len(objs) == 2
    assert all_usage == 36
    assert truncated == False

def test_concate_llm_contents_truncated():
    contents = [
        '[{"aaa": "...", "bbb": "...", "ccc": "..."}, {"aaa": "...", "bbb": ',
        '[{"aaa": "...", "bbb": "...", "ccc": "..."}]',
    ]
    all_contents, all_usage, truncated = concate_llm_contents(contents=contents, usages=[24, 12])
    objs = json.loads(all_contents)
    assert len(objs) == 2
    assert all_usage == 36
    assert truncated == True

def test_concate_llm_contents_error_handling():
    with open("tests/data/36396314_gpt_4o_error_result.json") as fobj:
        obj = json.load(fobj)
        res, usage, truncated = concate_llm_contents(obj["contents"], [10000, 3000])
        assert truncated == True
        assert usage == 13000



