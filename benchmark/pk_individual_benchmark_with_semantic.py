from typing import Literal, Tuple, Union
import pandas as pd

from benchmark.evaluate import TablesEvaluator, TablesSeparateEvaluator
from benchmark.common import ColumnType

DRUG_NAME = "Drug name"
ANALYTE = "Analyte"
SPECIMEN = "Specimen"
POPULATION = "Population"
PREGNANCY_STAGE = "Pregnancy stage"
PEDIATRIC_GESTATIONAL_AGE = "Pediatric/Gestational age"
PARAMETER_TYPE = "Parameter type"
PARAMETER_UNIT = "Parameter unit"
PARAMETER_VALUE = "Parameter value"
TIME_VALUE = "Time value"
TIME_UNIT = "Time unit"
PATIENT_ID = "Patient ID"

PK_INDIVIDUAL_COLUMNS_TYPE = {
    "Drug name": ColumnType.Text,
    "Analyte": ColumnType.Text,
    "Specimen": ColumnType.Text,
    "Population": ColumnType.Text,
    "Pregnancy stage": ColumnType.Text,
    "Pediatric/Gestational age": ColumnType.Text,
    "Parameter type": ColumnType.Text,
    "Parameter unit": ColumnType.Text,
    "Parameter value": ColumnType.Numeric,
    "Time value": ColumnType.Numeric,
    "Time unit": ColumnType.Text,
}

PK_INDIVIDUAL_RATING_COLUMNS = [
    (DRUG_NAME, 2.0),
    (ANALYTE, 2.0),
    (SPECIMEN, 0.5),
    (POPULATION, 0.5),
    (PREGNANCY_STAGE, 0.5),
    (PEDIATRIC_GESTATIONAL_AGE, 0.5),
    (PARAMETER_TYPE, 1.0),
    (PARAMETER_UNIT, 1.0),
    (PARAMETER_VALUE, 5.0),
]

PK_INDIVIDUAL_ANCHOR_COLUMNS = [
    PATIENT_ID,
    PARAMETER_VALUE,
    DRUG_NAME,
    ANALYTE,
    TIME_VALUE,
]

def pk_individual_evaluate_dataframe(
    df_baseline: pd.DataFrame,
    df: pd.DataFrame,
    score_mode: Literal["combined", "separate"] | None = "combined",
) -> int | Tuple[int, int]:
    evaluator = TablesEvaluator(
        rating_cols=PK_INDIVIDUAL_RATING_COLUMNS,
        anchor_cols=PK_INDIVIDUAL_ANCHOR_COLUMNS,
        columns_type=PK_INDIVIDUAL_COLUMNS_TYPE,
    ) if score_mode == "combined" else TablesSeparateEvaluator(
        rating_cols=PK_INDIVIDUAL_RATING_COLUMNS,
        anchor_cols=PK_INDIVIDUAL_ANCHOR_COLUMNS,
        columns_type=PK_INDIVIDUAL_COLUMNS_TYPE,
    )
    return evaluator.compare_tables(df_baseline, df)