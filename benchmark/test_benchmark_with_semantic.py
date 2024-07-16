import pandas as pd
import pytest

from benchmark.evaluate import compare_tables
from benchmark.common import output_msg

@pytest.mark.skip()
@pytest.mark.parametrize("pmid,expected", [
    ("35489632", 80), 
    ("30825333", 50), 
    ("16143486", 67), 
    ("22050870", 50), 
    ("17635501", 81),])
def test_gemini15(pmid, expected):
    df_target = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-gemini15.csv")
    df_baseline = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-baseline.csv")
    
    score = compare_tables(df_baseline, df_target)
    assert score == expected

# @pytest.mark.skip()
@pytest.mark.parametrize("pmid,expected", [
    ("35489632", 80), 
    ("30825333", 82), 
    ("16143486", 67), 
    ("22050870", 52), 
    ("17635501", 82)])
def test_gpt4o(pmid, expected):
    df_target = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-gpt4o.csv")
    df_baseline = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-baseline.csv")
    
    score = compare_tables(df_baseline, df_target)
    assert score == expected


@pytest.mark.skip()
@pytest.mark.parametrize("pmid, expected", [
    ("29943508", 80),
    ("30950674", 80),
    ("34114632", 80),
    ("34183327", 80),
    ("35465728", 80),
])
def test_5_papers(pmid, expected):
    df_gemini = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-gemini15.csv")
    df_gpt4o = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-gpt4o.csv")
    df_baseline = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-baseline.csv")

    score = compare_tables(df_baseline, df_gemini)
    output_msg(f"{pmid} gemini score: {score}")
    # assert score == expected
    
    scroe = compare_tables(df_baseline, df_gpt4o)
    output_msg(f"{pmid} gpt4o score: {score}")
    assert score == expected

