import pytest
import os
from typing import List, Any
import csv
import pandas as pd
from pandas import DataFrame, Series
import math
import functools

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
]

def compare_row(baseline: List[Any], target: List[Any]) -> int:
    pass

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

    return abs(sum1 - sum2) < 0.000001

"""
    for c in ANCHOR_COLUMNS:
        v1 = row1[c]
        v2 = row2[c]
        v1 = v1.strip() if isinstance(v1, str) else v1
        v2 = v2.strip() if isinstance(v2, str) else v2
        if v1 != v2:
            if math.isnan(v1) and math.isnan(v1):
                continue
            return False
        
    return True
"""

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
        v1 = v1.strip() if isinstance(v1, str) else v1
        v2 = v2.strip() if isinstance(v2, str) else v2
        
        if v1 == v2:
            sum +=1
        elif isinstance(v1, str) or isinstance(v2, str):
            continue
        elif math.isnan(v1) and math.isnan(v2):
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



