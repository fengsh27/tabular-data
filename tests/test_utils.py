
import pytest

from extractor.utils import (
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

