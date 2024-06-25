import pytest
import os
from typing import List, Any
import csv
import pandas as pd
from pandas import DataFrame, Series
import math
import functools
import logging

logger = logging.getLogger(__name__)

"""
row count:      10/100
Parameter type: 10/100
Value:          20/100 
Unit:           10/100  
Time:           10/100 
Time unit:      10/100 
Subjects:       15/100 
Drug Name:      15/100 
"""

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

RATING_COLUMNS = [
    DRUG_NAME,
    PARAMETER_TYPE,
    VALUE,
    UNIT,
    SUBJECTS,
    VARIATION_TYPE,
    VARIATION_VALUE,
    P_VALUE,
]
ANCHOR_COLUMNS = [
    VALUE,
    VARIATION_VALUE,
    LOWER_LIMIT,
    HIGH_LIMIT,
    P_VALUE,
]
DELTA_VALUE = 0.000001

def parse_value(val: Any) -> (int | float | None):
    if isinstance(val, (int, float)) and not math.isnan(val):
        return val
    if not isinstance(val, str):
        return None
    try: 
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return None

def anchor_row(row1: Series, row2: Series):
    """
    # locate row by sum value
    sum1 = 0
    for index, value in row1.items():
        if isinstance(index, str) and index.startswith("Unnamed"):
            continue
        val = parse_value(value)
        if val is None:
            continue
        sum1 += val
    sum2 = 0
    for index, value in row2.items():
        if isinstance(index, str) and index.startswith("Unnamed"):
            continue
        val = parse_value(value)
        if val is None:
            continue
        sum2 += val

    return abs(sum1 - sum2) < DELTA_VALUE
    """
    # locate row by ANCHOR_ROWS
    for c in ANCHOR_COLUMNS:
        v1 = row1[c]
        v2 = row2[c]
        if not compare_value(v1, v2):
            return False        
    return True


def compare_value(v1, v2) -> bool:
    if v1 == v2:
        return True
    if isinstance(v1, str) and isinstance(v2, str):
        return v1.strip().lower() == v2.strip().lower()
    if not isinstance(v1, str) and not isinstance(v2, str):
        if math.isnan(v1) and math.isnan(v2):
            return True
        fv1 = float(v1)
        fv2 = float(v2)
        return abs(fv1 - fv2) < DELTA_VALUE
    
    # One of the two values is str
    sval = v1 if isinstance(v1, str) else v2
    nsval = v1 if not isinstance(v1, str) else v2
    if math.isnan(nsval) and len(sval.strip()) == 0:
        return True

    try:
        fv1 = float(v1)
        fv2 = float(v2)
        return abs(fv1 - fv2) < DELTA_VALUE
    except Exception as e:
        logger.error(e)
        return False
        

def anchor_row_from_dataframe(row: Series, df: DataFrame):
    for ix, r in df.iterrows():
        if anchor_row(r, row):
            return r
    return None

def rate_row(row1: Series, row2: Series) -> int:
    sum = 0
    for c in RATING_COLUMNS:
        v1 = row1[c]
        v2 = row2[c]
        
        if compare_value(v1, v2):
            sum += 1

    return (int)(10.0 * sum / float(len(RATING_COLUMNS)))

def rate_rows(baseline: DataFrame, target: DataFrame) -> int:
    bshape = baseline.shape
    tshape = target.shape
    if bshape[1] != tshape[1]:
        return 0
    
    less = baseline if bshape[0] <= tshape[0] else target
    much = baseline if bshape[0] > tshape[0] else target
    less_row_num = less.shape[0]
    much_row_num = much.shape[0]
    scores = []
    sum = 0
    for index, row in less.iterrows():
        much_row = anchor_row_from_dataframe(row, much)
        if much_row is None:
            scores.append(0)
            continue
        scores.append(rate_row(row, much_row))
    sum = functools.reduce(lambda a, b: a+b, scores)
    
    return (int)(100.0 * (sum / (10.0 * much_row_num)))
    

def compare_tables(baseline:pd.DataFrame, target:pd.DataFrame) -> tuple[int, int]:
    return rate_rows(baseline, target)


