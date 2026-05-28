from dataclasses import dataclass
from typing import Callable, Optional, Union

import pandas as pd

from .common import ColumnType
from .constant import BenchmarkType
from .pe_preprocess import preprocess_table as preprocess_pe_table
from .pk_preprocess import (
    preprocess_pk_individual_table,
    preprocess_pk_summary_table,
    preprocess_pk_drug_summary_table,
    preprocess_pk_specimen_summary_table,
    preprocess_pk_population_summary_table,
    preprocess_pk_drug_individual_table,
    preprocess_pk_specimen_individual_table,
    preprocess_pk_population_individual_table,
)

# ── shared column names ────────────────────────────────────────────────────────
DRUG_NAME = "Drug name"
ANALYTE = "Analyte"
SPECIMEN = "Specimen"
POPULATION = "Population"
PEDIATRIC_GESTATIONAL_AGE = "Pediatric/Gestational age"
PREGNANCY_STAGE = "Pregnancy stage"
PARAMETER_TYPE = "Parameter type"
VALUE = "Value"
UNIT = "Unit"
P_VALUE = "P value"
LOWER_LIMIT = "Lower limit"
HIGH_LIMIT = "High limit"
LOWER_BOUND = "Lower bound"
UPPER_BOUND = "Upper bound"
TIME_VALUE = "Time value"
TIME_UNIT = "Time unit"
PATIENT_ID = "Patient ID"
NOTE = "Note"
PARAMETER_UNIT = "Parameter unit"
PARAMETER_VALUE = "Parameter value"
SUBJECT_N = "Subject N"
PATIENT_CHARACTERISTIC = "Patient characteristic"
CHARACTERISTIC_SUB_CATEGORY = "Characteristic sub-category"
MAIN_VALUE = "Main value"
SOURCE_TEXT = "Source text"

# ── PK summary ────────────────────────────────────────────────────────────────
SUMMARY_STATISTICS = "Summary statistics"
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
    SUBJECT_N,
    LOWER_LIMIT,
    HIGH_LIMIT,
    P_VALUE,
    DRUG_NAME,
    ANALYTE,
    SPECIMEN,
    POPULATION,
    PREGNANCY_STAGE,
]

# ── PK drug summary ─────────────────────────────────────────────────────────────
DRUG_METABOLITE_NAME = "Drug/Metabolite name"
DOSE_AMOUNT = "Dose amount"
DOSE_UNIT = "Dose unit"
DOSE_FREQUENCY = "Dose frequency"
DOSE_SCHEDULE = "Dose schedule"
DOSE_ROUTE = "Dose route"

PK_DRUG_SUMMARY_COLUMNS_TYPE: dict[str, ColumnType] = {
    DRUG_METABOLITE_NAME: ColumnType.Text,
    DOSE_AMOUNT: ColumnType.Text,
    DOSE_UNIT: ColumnType.Text,
    DOSE_FREQUENCY: ColumnType.Text,
    DOSE_SCHEDULE: ColumnType.Text,
    DOSE_ROUTE: ColumnType.Text,
    POPULATION: ColumnType.Text,
    PREGNANCY_STAGE: ColumnType.Text,
    PEDIATRIC_GESTATIONAL_AGE: ColumnType.Text,
    SUBJECT_N: ColumnType.Numeric,
    NOTE: ColumnType.Text,
}

PK_DRUG_SUMMARY_RATING_COLUMNS: list = [
    DRUG_METABOLITE_NAME,
    DOSE_AMOUNT,
    DOSE_UNIT,
    DOSE_FREQUENCY,
    DOSE_SCHEDULE,
    DOSE_ROUTE,
    POPULATION,
    PREGNANCY_STAGE,
    PEDIATRIC_GESTATIONAL_AGE,
    SUBJECT_N,
]
PK_DRUG_SUMMARY_ANCHOR_COLUMNS: list = [
    DRUG_METABOLITE_NAME,
    DOSE_AMOUNT,
    DOSE_UNIT,
    DOSE_FREQUENCY,
    SUBJECT_N,
    DOSE_SCHEDULE,
    DOSE_ROUTE,
    POPULATION,
    PREGNANCY_STAGE,
    PEDIATRIC_GESTATIONAL_AGE,    
]

# ── PK specimen summary ─────────────────────────────────────────────────────────────
SAMPLE_N = "Sample N"
SAMPLE_TIME = "Sample time"

PK_SPECIMEN_SUMMARY_COLUMNS_TYPE: dict[str, ColumnType] = {
    SPECIMEN: ColumnType.Text,
    SAMPLE_N: ColumnType.Numeric,
    POPULATION: ColumnType.Text,
    PREGNANCY_STAGE: ColumnType.Text,
    PEDIATRIC_GESTATIONAL_AGE: ColumnType.Text,
    SUBJECT_N: ColumnType.Numeric,
    SAMPLE_TIME: ColumnType.Text,
    TIME_UNIT: ColumnType.Text,
    NOTE: ColumnType.Text,
}

PK_SPECIMEN_SUMMARY_RATING_COLUMNS: list = [
    SPECIMEN,
    SAMPLE_N,
    POPULATION,
    PREGNANCY_STAGE,
    PEDIATRIC_GESTATIONAL_AGE,
    SUBJECT_N,
    SAMPLE_TIME,
    TIME_UNIT,
]
PK_SPECIMEN_SUMMARY_ANCHOR_COLUMNS: list = [
    SPECIMEN,
    SAMPLE_N,
    SUBJECT_N,
    POPULATION,
    PREGNANCY_STAGE,
    PEDIATRIC_GESTATIONAL_AGE,
    SAMPLE_TIME,
    TIME_UNIT,
]

# ── PK population summary ─────────────────────────────────────────────────────────────
CHARACTERISTIC = "Characteristic"
CHARACTERISTIC_SUBCATEGORY = "Characteristic subcategory"
CHARACTERISTIC_UNIT = "Characteristic unit"
CHARACTERISTIC_VALUE = "Characteristic value"

PK_POPULATION_SUMMARY_COLUMNS_TYPE: dict[str, ColumnType] = {
    PATIENT_ID: ColumnType.Text,
    CHARACTERISTIC: ColumnType.Text,
    CHARACTERISTIC_SUBCATEGORY: ColumnType.Text,
    CHARACTERISTIC_UNIT: ColumnType.Text,
    CHARACTERISTIC_VALUE: ColumnType.Numeric,
    POPULATION: ColumnType.Text,
    PREGNANCY_STAGE: ColumnType.Text,
    PEDIATRIC_GESTATIONAL_AGE: ColumnType.Text,
    NOTE: ColumnType.Text,
}

PK_POPULATION_SUMMARY_RATING_COLUMNS: list = [
    PATIENT_ID,
    CHARACTERISTIC_VALUE,
    CHARACTERISTIC,
    CHARACTERISTIC_SUBCATEGORY,
    CHARACTERISTIC_UNIT,
    POPULATION,
    PREGNANCY_STAGE,
    PEDIATRIC_GESTATIONAL_AGE,
]
PK_POPULATION_SUMMARY_ANCHOR_COLUMNS: list = [
    PATIENT_ID,
    CHARACTERISTIC,
    CHARACTERISTIC_SUBCATEGORY,
    CHARACTERISTIC_UNIT,
    CHARACTERISTIC_VALUE,
    POPULATION,
    PREGNANCY_STAGE,
    PEDIATRIC_GESTATIONAL_AGE,
]

# ── PK individual ─────────────────────────────────────────────────────────────
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

# ── PK drug individual ────────────────────────────────────────────────────────
# Columns: Patient ID, Drug/Metabolite name, Dose amount, Dose unit, Dose frequency,
#          Dose schedule, Dose route, Population, Pregnancy stage, Pediatric/Gestational age, Note

PK_DRUG_INDIVIDUAL_COLUMNS_TYPE: dict[str, ColumnType] = {
    PATIENT_ID: ColumnType.Text,
    DRUG_METABOLITE_NAME: ColumnType.Text,
    DOSE_AMOUNT: ColumnType.Numeric,
    DOSE_UNIT: ColumnType.Text,
    DOSE_FREQUENCY: ColumnType.Text,
    DOSE_SCHEDULE: ColumnType.Text,
    DOSE_ROUTE: ColumnType.Text,
    POPULATION: ColumnType.Text,
    PREGNANCY_STAGE: ColumnType.Text,
    PEDIATRIC_GESTATIONAL_AGE: ColumnType.Text,
    NOTE: ColumnType.Text,
}

PK_DRUG_INDIVIDUAL_RATING_COLUMNS: list = [
    DRUG_METABOLITE_NAME,
    DOSE_AMOUNT,
    DOSE_UNIT,
    DOSE_FREQUENCY,
    DOSE_SCHEDULE,
    DOSE_ROUTE,
    POPULATION,
    PREGNANCY_STAGE,
    PEDIATRIC_GESTATIONAL_AGE,
]
PK_DRUG_INDIVIDUAL_ANCHOR_COLUMNS: list = [
    PATIENT_ID,
    DRUG_METABOLITE_NAME,
    DOSE_AMOUNT,
    DOSE_FREQUENCY,
    DOSE_SCHEDULE,
    DOSE_ROUTE,
]

# ── PK specimen individual ────────────────────────────────────────────────────
# Columns: Patient ID, Specimen, Sample N, Population, Pregnancy stage,
#          Pediatric/Gestational age, Sample time, Time unit, Note

PK_SPECIMEN_INDIVIDUAL_COLUMNS_TYPE: dict[str, ColumnType] = {
    PATIENT_ID: ColumnType.Text,
    SPECIMEN: ColumnType.Text,
    SAMPLE_N: ColumnType.Numeric,
    POPULATION: ColumnType.Text,
    PREGNANCY_STAGE: ColumnType.Text,
    PEDIATRIC_GESTATIONAL_AGE: ColumnType.Text,
    SAMPLE_TIME: ColumnType.Text,
    TIME_UNIT: ColumnType.Text,
    NOTE: ColumnType.Text,
}

PK_SPECIMEN_INDIVIDUAL_RATING_COLUMNS: list = [
    SPECIMEN,
    SAMPLE_N,
    POPULATION,
    PREGNANCY_STAGE,
    PEDIATRIC_GESTATIONAL_AGE,
    SAMPLE_TIME,
    TIME_UNIT,
]
PK_SPECIMEN_INDIVIDUAL_ANCHOR_COLUMNS: list = [
    PATIENT_ID,
    SPECIMEN,
    SAMPLE_N,
    POPULATION,
]

# ── PK population individual ──────────────────────────────────────────────────
# Columns: Patient ID, Patient characteristic, Characteristic sub-category,
#          Unit, Main value, Population, Pregnancy stage, Pediatric/Gestational age, Source text

PK_POPULATION_INDIVIDUAL_COLUMNS_TYPE: dict[str, ColumnType] = {
    PATIENT_ID: ColumnType.Text,
    PATIENT_CHARACTERISTIC: ColumnType.Text,
    CHARACTERISTIC_SUB_CATEGORY: ColumnType.Text,
    UNIT: ColumnType.Text,
    MAIN_VALUE: ColumnType.Numeric,
    POPULATION: ColumnType.Text,
    PREGNANCY_STAGE: ColumnType.Text,
    PEDIATRIC_GESTATIONAL_AGE: ColumnType.Text,
    SOURCE_TEXT: ColumnType.Text,
}

PK_POPULATION_INDIVIDUAL_RATING_COLUMNS: list = [
    PATIENT_CHARACTERISTIC,
    CHARACTERISTIC_SUB_CATEGORY,
    UNIT,
    MAIN_VALUE,
    POPULATION,
    PREGNANCY_STAGE,
    PEDIATRIC_GESTATIONAL_AGE,
]
PK_POPULATION_INDIVIDUAL_ANCHOR_COLUMNS: list = [
    PATIENT_ID,
    PATIENT_CHARACTERISTIC,
    CHARACTERISTIC_SUB_CATEGORY,
    MAIN_VALUE,
]

# ── PE study info ─────────────────────────────────────────────────────────────
# Study type,Population,Study design,Pregnancy stage,Drug name,Data source,Inclusion criteria,Exclusion criteria,Outcomes,Subject N
STUDY_TYPE = "Study type"
STUDY_DESIGN = "Study design"
DATA_SOURCE = "Data source"
INCLUSION_CRITERIA = "Inclusion criteria"
EXCLUSION_CRITERIA = "Exclusion criteria"
OUTCOMES = "Outcomes"

PE_STUDY_INFO_COLUMNS_TYPE: dict[str, ColumnType] = {
    STUDY_TYPE: ColumnType.Text,
    POPULATION: ColumnType.Text,
    STUDY_DESIGN: ColumnType.Text,
    PREGNANCY_STAGE: ColumnType.Text,
    DRUG_NAME: ColumnType.Text,
    DATA_SOURCE: ColumnType.Text,
    INCLUSION_CRITERIA: ColumnType.Text,
    EXCLUSION_CRITERIA: ColumnType.Text,
    OUTCOMES: ColumnType.Text,
    SUBJECT_N: ColumnType.Numeric,
}

PE_STUDY_INFO_RATING_COLUMNS: list = [
    STUDY_TYPE,
    STUDY_DESIGN,
    DATA_SOURCE,
    INCLUSION_CRITERIA,
    EXCLUSION_CRITERIA,
    OUTCOMES,
    SUBJECT_N,
]
PE_STUDY_INFO_ANCHOR_COLUMNS: list = [
    SUBJECT_N,
    STUDY_TYPE,
    STUDY_DESIGN,
    DATA_SOURCE,
    INCLUSION_CRITERIA,
    EXCLUSION_CRITERIA,
    OUTCOMES,
]

# ── PE study outcome ────────────────────────────────────────────────────────────────────────
EXPOSURE = "Exposure"
OUTCOMES = "Outcomes"
PARAMETER_STATISTIC = "Parameter statistic"
# VARIABILITY_VALUE = "Variability value"

PE_STUDY_OUTCOME_COLUMNS_TYPE: dict[str, ColumnType] = {
    CHARACTERISTIC: ColumnType.Text,
    EXPOSURE: ColumnType.Text,
    OUTCOMES: ColumnType.Text,
    PARAMETER_STATISTIC: ColumnType.Text,
    PARAMETER_VALUE: ColumnType.Numeric,
    PARAMETER_UNIT: ColumnType.Text,
    # VARIABILITY_VALUE: ColumnType.Numeric,
    VARIATION_TYPE: ColumnType.Text,
    VARIATION_VALUE: ColumnType.Numeric,
    LOWER_BOUND: ColumnType.Numeric,
    UPPER_BOUND: ColumnType.Numeric,
}

PE_STUDY_OUTCOME_RATING_COLUMNS: list = [
    EXPOSURE,
    OUTCOMES,
    PARAMETER_STATISTIC,
    PARAMETER_VALUE,
    PARAMETER_UNIT,
    VARIATION_TYPE,
    VARIATION_VALUE,
    LOWER_BOUND,
    UPPER_BOUND,
]
PE_STUDY_OUTCOME_ANCHOR_COLUMNS: list = [
    PARAMETER_VALUE,
    LOWER_BOUND,
    UPPER_BOUND,
    P_VALUE,
    VARIATION_VALUE,
    EXPOSURE,
    OUTCOMES,
    CHARACTERISTIC,
    PARAMETER_STATISTIC,
    PARAMETER_UNIT,
    VARIATION_TYPE,
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
    BenchmarkType.PE_STUDY_OUTCOME: BenchmarkConfig(
        benchmark_type=BenchmarkType.PE_STUDY_OUTCOME,
        preprocess=preprocess_pe_table,
        rating_cols=PE_STUDY_OUTCOME_RATING_COLUMNS,
        anchor_cols=PE_STUDY_OUTCOME_ANCHOR_COLUMNS,
        columns_type=PE_STUDY_OUTCOME_COLUMNS_TYPE,
    ),
    BenchmarkType.PE_STUDY_INFO: BenchmarkConfig(
        benchmark_type=BenchmarkType.PE_STUDY_INFO,
        preprocess=preprocess_pe_table,
        rating_cols=PE_STUDY_INFO_RATING_COLUMNS,
        anchor_cols=PE_STUDY_INFO_ANCHOR_COLUMNS,
        columns_type=PE_STUDY_INFO_COLUMNS_TYPE,
    ),
    BenchmarkType.PK_POPULATION_SUMMARY: BenchmarkConfig(
        benchmark_type=BenchmarkType.PK_POPULATION_SUMMARY,
        preprocess=preprocess_pk_population_summary_table,
        rating_cols=PK_POPULATION_SUMMARY_RATING_COLUMNS,
        anchor_cols=PK_POPULATION_SUMMARY_ANCHOR_COLUMNS,
        columns_type=PK_POPULATION_SUMMARY_COLUMNS_TYPE,
    ),
    BenchmarkType.PK_POPULATION_INDIVIDUAL: BenchmarkConfig(
        benchmark_type=BenchmarkType.PK_POPULATION_INDIVIDUAL,
        preprocess=preprocess_pk_population_individual_table,
        rating_cols=PK_POPULATION_INDIVIDUAL_RATING_COLUMNS,
        anchor_cols=PK_POPULATION_INDIVIDUAL_ANCHOR_COLUMNS,
        columns_type=PK_POPULATION_INDIVIDUAL_COLUMNS_TYPE,
    ),
    BenchmarkType.PK_SPECIMEN_INDIVIDUAL: BenchmarkConfig(
        benchmark_type=BenchmarkType.PK_SPECIMEN_INDIVIDUAL,
        preprocess=preprocess_pk_specimen_individual_table,
        rating_cols=PK_SPECIMEN_INDIVIDUAL_RATING_COLUMNS,
        anchor_cols=PK_SPECIMEN_INDIVIDUAL_ANCHOR_COLUMNS,
        columns_type=PK_SPECIMEN_INDIVIDUAL_COLUMNS_TYPE,
    ),
    BenchmarkType.PK_SPECIMEN_SUMMARY: BenchmarkConfig(
        benchmark_type=BenchmarkType.PK_SPECIMEN_SUMMARY,
        preprocess=preprocess_pk_specimen_summary_table,
        rating_cols=PK_SPECIMEN_SUMMARY_RATING_COLUMNS,
        anchor_cols=PK_SPECIMEN_SUMMARY_ANCHOR_COLUMNS,
        columns_type=PK_SPECIMEN_SUMMARY_COLUMNS_TYPE,
    ),
    BenchmarkType.PK_DRUG_INDIVIDUAL: BenchmarkConfig(
        benchmark_type=BenchmarkType.PK_DRUG_INDIVIDUAL,
        preprocess=preprocess_pk_drug_individual_table,
        rating_cols=PK_DRUG_INDIVIDUAL_RATING_COLUMNS,
        anchor_cols=PK_DRUG_INDIVIDUAL_ANCHOR_COLUMNS,
        columns_type=PK_DRUG_INDIVIDUAL_COLUMNS_TYPE,
    ),
    BenchmarkType.PK_DRUG_SUMMARY: BenchmarkConfig(
        benchmark_type=BenchmarkType.PK_DRUG_SUMMARY,
        preprocess=preprocess_pk_drug_summary_table,
        rating_cols=PK_DRUG_SUMMARY_RATING_COLUMNS,
        anchor_cols=PK_DRUG_SUMMARY_ANCHOR_COLUMNS,
        columns_type=PK_DRUG_SUMMARY_COLUMNS_TYPE,
    ),
}


def get_benchmark_config(benchmark_type: BenchmarkType) -> BenchmarkConfig:
    config = BENCHMARK_CONFIGS.get(benchmark_type)
    if config is None:
        raise ValueError(f"Unsupported benchmark type: {benchmark_type}")
    return config
