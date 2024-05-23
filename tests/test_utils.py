
import pytest

from extractor.utils import (
    preprocess_csv_table_string,
    convert_csv_table_to_dataframe,
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

    with open("./tests/17158945-result-1.txt", "r") as fobj:
        csv_str = fobj.read()
        table = convert_csv_table_to_dataframe(csv_str)
        assert table is not None
