import os
import shutil
from typing import List, Optional
import os.path as path
import csv
import pandas as pd

"""
This module is to pre-process tables generated by LLMs(ChatGPT/Gemini) to 
normalize them to facilitate benchmark.
"""

PK_COLUMNS = [
    "Drug name",
    "Analyte",
    "Specimen",
    "Population",
    "Pregnancy stage",
    "Summary statistics",
    "Parameter type",
    "Value",
    "Unit",
    "Subject N",
    "Variation value",
    "Variation type",
    "P value",
    "Interval type",
    "Lower limit",
    "High limit",
]
LOWER_PK_COLUMNS = [
    'drug name', 
    'analyte', 
    'specimen', 
    'population', 
    'pregnancy stage', 
    'summary statistics', 
    'parameter type', 
    'value', 
    'unit', 
    'subject n', 
    'variation value', 
    'variation type', 
    'p value', 
    'interval type', 
    'lower limit', 
    'high limit'
]
PK_COLUMNS_MAP = [
    ("P-value", "P value"),
    ("Subjectsn", "Subject N"),
    ("Subjects n", "Subject N"),
    ("Interval low", "Lower limit"),
    ("interval high", "High limit"),
    ("Variability statistic", "Variation type"),
    ('Summary Statistic', "Summary Statistics"),
    ('Parameter unit', "Unit"),
    ('Parameter statistic', "Summary Statistics"),
    ('Parameter value', "Value"),
    ("Lower bound", "Lower limit"),
    ("Upper bound", "High limit"),
]

def process_1st_column(rows: List[List[str]]):
    headers = rows[0]
    if headers[0].lower() == LOWER_PK_COLUMNS[0]:
        # no NO. column in table
        prc_rows = []
        for ix, row in enumerate(rows):
            prc_row = []
            if ix == 0:
                prc_row.append('')
            else:
                prc_row.append(ix-1)
            prc_row = prc_row + row
            prc_rows.append(prc_row)
        return prc_rows

    for ix, row in enumerate(rows):
        if ix == 0:
            row[0] = ""
        else:
            row[0] = ix - 1
    
    return rows

def process_column_names(rows: List[List[str]]):
    unknow_col_index = []

    for ix in range(len(rows[0])):
        column = rows[0][ix]
        if ix == 0:
            continue
        found = False
        if len(column.strip()) == 0:
            print(f"Error: the {ix}th column is empty")
            unknow_col_index.append(ix)
            continue
        # Normalize column names
        try:
            col_ix = LOWER_PK_COLUMNS.index(column.strip().lower())
            rows[0][ix] = PK_COLUMNS[col_ix]
            found = True
        except ValueError:
            found = False

        # Map column name
        if not found:
            pair = next(( x for x in PK_COLUMNS_MAP if x[0].lower() == column.strip().lower() ), None)
            if pair is not None:
                rows[0][ix] = pair[1]
                found = True
            # for pk_col_map in PK_COLUMNS_MAP:
            #     if column.strip().lower() == pk_col_map[0].lower():
            #         rows[0][ix] = pk_col_map[1]
            #         found = True
            #         break
        if not found:
            print(f"Error: can't find column {column}")
            unknow_col_index.append(ix)
    
    if len(unknow_col_index) == 0:
        return rows
    
    processed_rows = []
    for row in rows:
        prcssed_row = []
        for ix in range(len(row)):
            if ix not in unknow_col_index:
                prcssed_row.append(row[ix])
        processed_rows.append(prcssed_row)

    return processed_rows


def preprocess_table(csv_file) -> pd.DataFrame:
    with open(csv_file, "r") as fobj:
        reader = csv.reader(fobj)
        rows = list(reader)
        
        rows = process_1st_column(rows)
        rows = process_column_names(rows)

        output_df = pd.DataFrame(rows[1:], columns=rows[0])

        return output_df
    

def preprocess_PK_csv_file(pk_csv_file: str):
    bn, extname = path.splitext(pk_csv_file)
    orig_file = f"{bn}-original{extname}"
    try:
        if not os.path.exists(orig_file):
            shutil.copyfile(pk_csv_file, orig_file)
    except Exception as e:
        print(str(e))
        return False

    dst_file = pk_csv_file
    output_df = preprocess_table(orig_file)

    # before write to csv file, remove the first column,
    output_df = output_df.iloc[:, 1:]
    output_df.to_csv(dst_file, sep=',')
    return True

def preprocess_PK_csv_files(pk_csv_files: List[str]):
    for f in pk_csv_files:
        res = preprocess_PK_csv_file(f)
        if not res:
            print(f"Failed to pre-process file: {f}")

def process_single_file():
    import argparse
    from os import path
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="csv file path to be pre-processed")
    args = vars(parser.parse_args())
    print(args)
    preprocess_PK_csv_file(args["csv_file"])
    return


def process_multiple_files():
    import argparse
    from os import path
    
    pk_files = [
        "./benchmark/data/pk-summary/baseline/A/18394772_baseline.csv",
    ]
    preprocess_PK_csv_files(pk_files)
    
if __name__ == "__main__":
    process_single_file()

