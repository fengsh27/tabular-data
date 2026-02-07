import pytest
import pandas as pd

from benchmark.evaluate import TablesEvaluator, TablesSeparateEvaluator
from benchmark.pk_summary_benchmark_with_semantic import (
    PK_SUMMARY_ANCHOR_COLUMNS,
    PK_SUMMARY_RATING_COLUMNS,
    PK_SUMMARY_COLUMNS_TYPE,
)


@pytest.mark.skip("skip current due to expensive transformer installation")
def test_anchro_row_from_rows():
    pmid = "30950674"
    target = pd.read_csv(f"./benchmark/data/pk-summary/2024-10-16/{pmid}_gpt4o.csv")
    baseline = pd.read_csv(f"./benchmark/data/pk-summary/baseline/{pmid}_baseline.csv")

    bshape = baseline.shape
    tshape = target.shape
    if bshape[1] != tshape[1]:
        return 0

    less = baseline if bshape[0] <= tshape[0] else target
    much = baseline if bshape[0] > tshape[0] else target

    much_rows = much.to_dict("records")

    evaluator = TablesEvaluator(
        anchor_cols=PK_SUMMARY_ANCHOR_COLUMNS, rating_cols=PK_SUMMARY_RATING_COLUMNS
    )
    for i, r in less.iterrows():
        the_row = evaluator.anchor_row_from_rows(r, much_rows)
        assert the_row is not None

@pytest.mark.skip("skip current due to expensive transformer installation")
def test_compare_tables():
    pmid = "30950674"
    target = pd.read_csv(f"./benchmark/data/pk-summary/2024-10-16/{pmid}_gpt4o.csv")
    baseline = pd.read_csv(f"./benchmark/data/pk-summary/baseline/{pmid}_baseline.csv")

    bshape = baseline.shape
    tshape = target.shape
    if bshape[1] != tshape[1]:
        return 0
    evaluator = TablesSeparateEvaluator(
        anchor_cols=PK_SUMMARY_ANCHOR_COLUMNS,
        rating_cols=PK_SUMMARY_RATING_COLUMNS,
        columns_type=PK_SUMMARY_COLUMNS_TYPE,
    )

    text_score, numeric_score = evaluator.compare_tables(baseline, target)
    assert text_score > 0
    assert numeric_score > 0

    evaluator = TablesEvaluator(
        anchor_cols=PK_SUMMARY_ANCHOR_COLUMNS,
        rating_cols=PK_SUMMARY_RATING_COLUMNS,
    )
    score = evaluator.compare_tables(baseline, target)
    assert score > 0
