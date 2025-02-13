import pytest
import pandas as pd

from benchmark.evaluate import TablesEvaluator
from benchmark.test_benchmark_with_semantic import (
    PK_ANCHOR_COLUMNS, 
    PK_RATING_COLUMNS
)

@pytest.mark.skip("skip current due to expensive transformer installation")
def test_anchro_row_from_rows():
    pmid = "30950674"
    target = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-gpt4o.csv")
    baseline = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-baseline.csv")

    bshape = baseline.shape
    tshape = target.shape
    if bshape[1] != tshape[1]:
        return 0
        
    less = baseline if bshape[0] <= tshape[0] else target
    much = baseline if bshape[0] > tshape[0] else target
    
    much_rows = much.to_dict('records')
    
    evaluator = TablesEvaluator(
        anchor_cols=PK_ANCHOR_COLUMNS, 
        rating_cols=PK_RATING_COLUMNS
    )
    for i, r in less.iterrows():
        the_row = evaluator.anchor_row_from_rows(r, much_rows)
        assert the_row is not None


