import json
import pytest

from extractor.utils import (
    extract_float_value,
    preprocess_csv_table_string,
    remove_comma_in_number_string,
)

def test_preprocess_csv_table_string():
    with open("./tests/17158945-result.txt", "r") as fobj:
        csv_str = fobj.read()
        assert len(csv_str) == 2497
        out_str = preprocess_csv_table_string(csv_str)
        assert len(out_str) == 2479

    with open("./tests/32510456-result.txt", "r") as fobj:
        csv_str = fobj.read()
        cur_length = len(csv_str)
        out_str = preprocess_csv_table_string(csv_str)
        processed_length = len(out_str)
        assert cur_length == processed_length

@pytest.mark.parametrize("content, expected", [
    ("1,234.567", "1234.567"),
    ("-123,456.789", "-123456.789"),
    ("+123,456.789", "+123456.789"),
    (",123456", ",123456"),
    (",123,456", ",123456"),
])
def test_process_number_string(content, expected):
    processed_content = remove_comma_in_number_string(content)
    assert processed_content == expected


def test_json_loads():
    str1 = '[{"aaa": "...", "bbb": "...", "ccc": "..."}, {"aaa": "...", "bbb": '
    try:
        obj1 = json.loads(str1)
        print(obj1)
    except json.JSONDecodeError as e:
        print(e)

def test_extract_float_value():
    str1 = "*1.234"
    str2 = "*-1.234"
    str3 = "1.234*"
    str4 = "-1.234*"
    str5 = "1.234"
    v1 = extract_float_value(str1)
    assert v1 == 1.234
    v2 = extract_float_value(str2)
    assert v2 == -1.234
    v3 = extract_float_value(str3)
    assert v3 == 1.234
    v4 = extract_float_value(str4)
    assert v4 == -1.234
    v5 = extract_float_value(str5)
    assert v5 == 1.234

def test_single_html_to_markdown(md_table_aligned_29943508):
    import re
    replaced_content: str = re.sub(r'\xa0', ' ', md_table_aligned_29943508)
    assert replaced_content.find("\xa0") < 0