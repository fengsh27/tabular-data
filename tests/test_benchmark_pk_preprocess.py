import pytest
import os

from benchmark.pk_preprocess import (
    ensure_NO_column, 
    ensure_columns,
    preprocess_pk_summary_table,
    PK_SUMMARY_COLUMNS,
)

def test_preprocess_pk_summary_table():
    csv_files = []
    for root, _, files in os.walk("tests/data/csv"):
        csv_files = [os.path.join(root, f) for f in files]

    for f in csv_files:
        df_table = preprocess_pk_summary_table(f)
        assert df_table.shape[1] == len(PK_SUMMARY_COLUMNS) + 1






