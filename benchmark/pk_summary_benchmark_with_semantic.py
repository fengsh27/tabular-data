import pandas as pd

from benchmark.evaluate import (
    TablesEvaluator,
)

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

def pk_summary_evaluate_dataframe(df_baseline: pd.DataFrame, df_pk_summary: pd.DataFrame) -> int:
    evaluator = TablesEvaluator(
        rating_cols=PK_RATING_COLUMNS,
        anchor_cols=PK_ANCHOR_COLUMNS,
    )
    return evaluator.compare_tables(df_baseline, df_pk_summary)

def pk_summary_evaluate_csvfile(baseline_fn: str, pk_summary_fn: str) -> int:
    df_baseline = pd.read_csv(baseline_fn)
    df_pk_summary = pd.read_csv(pk_summary_fn)
    return pk_summary_evaluate_dataframe(df_baseline, df_pk_summary)


