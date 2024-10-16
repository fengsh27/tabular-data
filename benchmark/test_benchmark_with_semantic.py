import pandas as pd
import pytest

from benchmark.evaluate import (
    TablesEvaluator,
)
from benchmark.common import output_msg

DRUG_NAME = "Drug name"
PARAMETER_TYPE="Parameter type"
VALUE="Value"
UNIT="Unit"
SUBJECTS="Subject N"
VARIATION_VALUE="Variation value"
VARIATION_TYPE="Variation type"
P_VALUE="P value"
INTERVAL_TYPE="Interval type"
LOWER_LIMIT="Lower limit"
HIGH_LIMIT="High limit"

PK_RATING_COLUMNS = [
    DRUG_NAME,
    PARAMETER_TYPE,
    VALUE,
    UNIT,
    SUBJECTS,
    VARIATION_TYPE,
    VARIATION_VALUE,
    P_VALUE,
]
PK_ANCHOR_COLUMNS = [
    VALUE,
    VARIATION_VALUE,
    LOWER_LIMIT,
    HIGH_LIMIT,
    P_VALUE,
]
PE_RATING_COLUMNS = [
    # "Characteristic/risk factor",
    "Exposure",
    "Outcomes",
    "Statistic",
    "Value",
    "Unit",
    "Variability value",
    LOWER_LIMIT,
    HIGH_LIMIT,
]
PE_ANCHOR_COLUMNS = [
    VALUE,
    LOWER_LIMIT,
    HIGH_LIMIT,
    P_VALUE,
    "Variability value",
]

@pytest.mark.parametrize("pmid, expected", [
    ("15930210", 16),
    ("18782787", 62),
    ("30308427", 10),
    ("33864754", 7),
    ("34024233", 14),
    ("34083820", 28),
    ("34741059", 0),
    ("35296792", 9),
    ("35997979", 0),
    ("36396314", 0),
])
def test_pe_gemini(pmid, expected):
    df_gemini = pd.read_csv(f"./benchmark/pe/{pmid}_gemini15.csv")
    df_baseline = pd.read_csv(f"./benchmark/pe/{pmid}_baseline.csv")

    evaluator = TablesEvaluator(
        rating_cols=PE_RATING_COLUMNS,
        anchor_cols=PE_ANCHOR_COLUMNS,
    )
    score = evaluator.compare_tables(df_baseline, df_gemini)
    output_msg(f"{pmid} gemini score: {score}")
    assert score == expected
    
@pytest.mark.parametrize("pmid, expected", [
    ("15930210", 48),
    ("18782787", 71),
    ("30308427", 22),
    ("33864754", 2),
    ("34024233", 43),
    ("34083820", 0),
    ("34741059", 0),
    ("35296792", 9),
    ("35997979", 0),
    ("36396314", 44),
])
def test_pe_gpt(pmid, expected):
    df_gpt4o = pd.read_csv(f"./benchmark/pe/{pmid}_gpt4o.csv")
    df_baseline = pd.read_csv(f"./benchmark/pe/{pmid}_baseline.csv")

    evaluator = TablesEvaluator(
        rating_cols=PE_RATING_COLUMNS,
        anchor_cols=PE_ANCHOR_COLUMNS,
    )
    
    score = evaluator.compare_tables(df_baseline, df_gpt4o)
    output_msg(f"{pmid} gpt4o score: {score}")
    assert score == expected

@pytest.mark.parametrize("pmid, expected", [
    ("29943508", 44),
    ("30950674", 0),
    ("34114632", 4),
    ("34183327", 47),
    ("35465728", 23),
    ("35489632", 100), 
    ("30825333", 91), 
    ("16143486", 0), 
    ("22050870", 24), 
    ("17635501", 85),
])
def test_pk_gemini(pmid, expected):
    df_gemini = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-gemini15.csv")
    df_baseline = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-baseline.csv")

    evaluator = TablesEvaluator(
        rating_cols=PK_RATING_COLUMNS,
        anchor_cols=PK_ANCHOR_COLUMNS,
    )
    score = evaluator.compare_tables(df_baseline, df_gemini)
    output_msg(f"{pmid} gemini score: {score}")
    assert score == expected
    

@pytest.mark.parametrize("pmid, expected", [
    ("29943508", 0),
    ("30950674", 0),
    ("34114632", 23),
    ("34183327", 63),
    ("35465728", 38),
    ("35489632", 80), 
    ("30825333", 91), 
    ("16143486", 80), 
    ("22050870", 70), 
    ("17635501", 82),
])
def test_pk_gpt(pmid, expected):
    df_gpt4o = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-gpt4o.csv")
    df_baseline = pd.read_csv(f"./benchmark/data/{pmid}-pk-summary-baseline.csv")

    evaluator = TablesEvaluator(
        rating_cols=PK_RATING_COLUMNS,
        anchor_cols=PK_ANCHOR_COLUMNS,
    )
    
    score = evaluator.compare_tables(df_baseline, df_gpt4o)
    output_msg(f"{pmid} gpt4o score: {score}")
    assert score == expected




