import pandas as pd
import pytest

from benchmark.evaluate import compare_tables

@pytest.mark.parametrize("pmid,expected", [("35489632", 100),])
def test_gemini15(pmid, expected):
    df_target = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-gemini15.csv")
    df_baseline = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-baseline.csv")
    
    score = compare_tables(df_baseline, df_target)
    assert score == expected

@pytest.mark.parametrize("pmid,expected", [("35489632", 80), ("30825333", 50), ("16143486", 67), ("22050870", 50), ("17635501", 81)])
def test_gpt4o(pmid, expected):
    df_target = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-gpt4o.csv")
    df_baseline = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-baseline.csv")
    
    score = compare_tables(df_baseline, df_target)
    assert score == expected


