import pandas as pd

from benchmark.evaluate import (
    TablesEvaluator,
)

VALUE="Value"
UNIT="Unit"
P_VALUE="P value"
LOWER_LIMIT="Lower limit"
HIGH_LIMIT="High limit"

PE_RATING_COLUMNS = [
    # "Characteristic/risk factor",
    "Exposure",
    "Outcomes",
    "Statistic",
    VALUE,
    UNIT,
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

def pe_summary_evaluate_dataframe(df_baseline: pd.DataFrame, df_pk_summary: pd.DataFrame) -> int:
    evaluator = TablesEvaluator(
        rating_cols=PE_RATING_COLUMNS,
        anchor_cols=PE_ANCHOR_COLUMNS,
    )
    return evaluator.compare_tables(df_baseline, df_pk_summary)

def pe_summary_evaluate_csvfile(baseline_fn: str, pk_summary_fn: str) -> int:
    df_baseline = pd.read_csv(baseline_fn)
    df_pk_summary = pd.read_csv(pk_summary_fn)
    return pe_summary_evaluate_dataframe(df_baseline, df_pk_summary)


