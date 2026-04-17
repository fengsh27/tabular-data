from dataclasses import dataclass
from typing import Callable, Optional, Union

import pandas as pd

from .common import ColumnType
from .constant import BenchmarkType
from .pe_preprocess import preprocess_table as preprocess_pe_table
from .pk_preprocess import (
    preprocess_pk_individual_table,
    preprocess_pk_summary_table,
)

# ── shared column names ────────────────────────────────────────────────────────
DRUG_NAME = "Drug name"
ANALYTE = "Analyte"
SPECIMEN = "Specimen"
POPULATION = "Population"
PREGNANCY_STAGE = "Pregnancy stage"
PARAMETER_TYPE = "Parameter type"
VALUE = "Value"
UNIT = "Unit"
P_VALUE = "P value"
LOWER_LIMIT = "Lower limit"
HIGH_LIMIT = "High limit"

# ── PK summary ────────────────────────────────────────────────────────────────
SUMMARY_STATISTICS = "Summary statistics"
SUBJECT_N = "Subject N"
VARIATION_VALUE = "Variation value"
VARIATION_TYPE = "Variation type"
INTERVAL_TYPE = "Interval type"

PK_SUMMARY_COLUMNS_TYPE: dict[str, ColumnType] = {
    DRUG_NAME: ColumnType.Text,
    ANALYTE: ColumnType.Text,
    SPECIMEN: ColumnType.Text,
    POPULATION: ColumnType.Text,
    PREGNANCY_STAGE: ColumnType.Text,
    SUMMARY_STATISTICS: ColumnType.Text,
    PARAMETER_TYPE: ColumnType.Text,
    VALUE: ColumnType.Numeric,
    UNIT: ColumnType.Text,
    SUBJECT_N: ColumnType.Numeric,
    VARIATION_VALUE: ColumnType.Numeric,
    VARIATION_TYPE: ColumnType.Text,
    P_VALUE: ColumnType.Numeric,
    INTERVAL_TYPE: ColumnType.Text,
    LOWER_LIMIT: ColumnType.Numeric,
    HIGH_LIMIT: ColumnType.Numeric,
}

PK_SUMMARY_RATING_COLUMNS: list = [
    DRUG_NAME,
    PARAMETER_TYPE,
    VALUE,
    UNIT,
    SUBJECT_N,
    VARIATION_TYPE,
    VARIATION_VALUE,
    P_VALUE,
]
PK_SUMMARY_ANCHOR_COLUMNS: list = [
    VALUE,
    VARIATION_VALUE,
    LOWER_LIMIT,
    HIGH_LIMIT,
    P_VALUE,
]

# ── PK individual ─────────────────────────────────────────────────────────────
PATIENT_ID = "Patient ID"
PEDIATRIC_GESTATIONAL_AGE = "Pediatric/Gestational age"
PARAMETER_UNIT = "Parameter unit"
PARAMETER_VALUE = "Parameter value"
TIME_VALUE = "Time value"
TIME_UNIT = "Time unit"

PK_INDIVIDUAL_COLUMNS_TYPE: dict[str, ColumnType] = {
    DRUG_NAME: ColumnType.Text,
    ANALYTE: ColumnType.Text,
    SPECIMEN: ColumnType.Text,
    POPULATION: ColumnType.Text,
    PREGNANCY_STAGE: ColumnType.Text,
    PEDIATRIC_GESTATIONAL_AGE: ColumnType.Text,
    PARAMETER_TYPE: ColumnType.Text,
    PARAMETER_UNIT: ColumnType.Text,
    PARAMETER_VALUE: ColumnType.Numeric,
    TIME_VALUE: ColumnType.Numeric,
    TIME_UNIT: ColumnType.Text,
}

PK_INDIVIDUAL_RATING_COLUMNS: list = [
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
PK_INDIVIDUAL_ANCHOR_COLUMNS: list = [
    PATIENT_ID,
    PARAMETER_VALUE,
    DRUG_NAME,
    ANALYTE,
    TIME_VALUE,
]

# ── PE ────────────────────────────────────────────────────────────────────────
EXPOSURE = "Exposure"
OUTCOMES = "Outcomes"
STATISTIC = "Statistic"
VARIABILITY_VALUE = "Variability value"

PE_RATING_COLUMNS: list = [
    EXPOSURE,
    OUTCOMES,
    STATISTIC,
    VALUE,
    UNIT,
    VARIABILITY_VALUE,
    LOWER_LIMIT,
    HIGH_LIMIT,
]
PE_ANCHOR_COLUMNS: list = [
    VALUE,
    LOWER_LIMIT,
    HIGH_LIMIT,
    P_VALUE,
    VARIABILITY_VALUE,
]


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_type: BenchmarkType
    preprocess: Callable[[str], pd.DataFrame]
    rating_cols: list
    anchor_cols: list
    columns_type: Optional[dict[str, ColumnType]] = None


BENCHMARK_CONFIGS: dict[BenchmarkType, BenchmarkConfig] = {
    BenchmarkType.PK_SUMMARY: BenchmarkConfig(
        benchmark_type=BenchmarkType.PK_SUMMARY,
        preprocess=preprocess_pk_summary_table,
        rating_cols=PK_SUMMARY_RATING_COLUMNS,
        anchor_cols=PK_SUMMARY_ANCHOR_COLUMNS,
        columns_type=PK_SUMMARY_COLUMNS_TYPE,
    ),
    BenchmarkType.PK_INDIVIDUAL: BenchmarkConfig(
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
        preprocess=preprocess_pk_individual_table,
        rating_cols=PK_INDIVIDUAL_RATING_COLUMNS,
        anchor_cols=PK_INDIVIDUAL_ANCHOR_COLUMNS,
        columns_type=PK_INDIVIDUAL_COLUMNS_TYPE,
    ),
    BenchmarkType.PE: BenchmarkConfig(
        benchmark_type=BenchmarkType.PE,
        preprocess=preprocess_pe_table,
        rating_cols=PE_RATING_COLUMNS,
        anchor_cols=PE_ANCHOR_COLUMNS,
        # columns_type intentionally omitted — preserves legacy PE behavior
        # (see refactor item 8).
    ),
}


def get_benchmark_config(benchmark_type: BenchmarkType) -> BenchmarkConfig:
    config = BENCHMARK_CONFIGS.get(benchmark_type)
    if config is None:
        raise ValueError(f"Unsupported benchmark type: {benchmark_type}")
    return config
