
import pytest
from extractor.generated_table_processor import GeneratedPKSummaryTableProcessor
from io import StringIO
import pandas as pd
import csv

from extractor.utils import convert_csv_table_to_dataframe
from extractor.constants import PKSUMMARY_TABLE_OUTPUT_COLUMNS

@pytest.mark.skip("csv parser can't handle comma in value correctly (have tried many ways)")
def test_pandas_read_csv():
    csv_header = ', '.join(PKSUMMARY_TABLE_OUTPUT_COLUMNS)
    csv_value1 = "lorazepam, Lorazepam enantiomeric mixture, blood, maternal, N/A, 8, concentration, 9.91, ng/ml, mean, None, None, 95% CI, 7.68, 12.14, None"
    csv_value2 = 'lorazepam, Lorazepam enantiomeric mixture, `cord blood,blood`, `neonate,maternal`, N/A, 8, cord blood/maternal blood, 0.73, None, mean, None, None, 95% CI, 0.52, 0.94, None'
    csv_str = f'{csv_header}\n{csv_value1}\n{csv_value2}\n'
    # buf = StringIO(csv_str)
    df = convert_csv_table_to_dataframe(csv_str)
    assert df is not None
    for i in range(16):
        the_val = df.iloc[0, i]
        print(the_val)
    

@pytest.mark.parametrize("pmid, expected", [
    ("29943508", (6,16)),
    ("34183327", (30, 16)),
    ("30950674_gemini", (30, 16)),
    ("34114632", (24, 16)),
])
def test_converter(pmid, expected):
    with open(f"./tests/{pmid}_result.json", "r") as fobj:
        res_str = fobj.read()
        processor = GeneratedPKSummaryTableProcessor()
        csv_str = processor.process_content(res_str)
        buf = StringIO(csv_str)
        df = pd.read_csv(buf)
        assert df is not None
        assert tuple(df.shape) == expected

def test_16143486_gemini_result_json():
    with open("./tests/16143486_gemini_result.json", "r") as fobj:
        res_str = fobj.read()
        processor = GeneratedPKSummaryTableProcessor()
        csv_str = processor.process_content(res_str)
        buf = StringIO(csv_str)
        df = pd.read_csv(buf)
        assert df is not None
        assert tuple(df.shape) == (29, 16)

def test_16143486_gemini_result_json_1():
    with open("./tests/16143486_gemini_result_1.json", "r") as fobj:
        res_str = fobj.read()
        processor = GeneratedPKSummaryTableProcessor()
        csv_str = processor.process_content(res_str)
        buf = StringIO(csv_str)
        df = pd.read_csv(buf)
        assert df is not None
        assert tuple(df.shape) == (30, 16)

def test_strip_table_content():
    str1 = "```json\n balahbalah ... \n```"
    processor = GeneratedPKSummaryTableProcessor()
    res = processor._strip_table_content(str1)
    assert res == "balahbalah ..."

    str11 = "```csv\n balahbalah ... \n```"
    processor = GeneratedPKSummaryTableProcessor()
    res = processor._strip_table_content(str11)
    assert res == "balahbalah ..."

    str12 = "\n balahbalah ... \n```"
    processor = GeneratedPKSummaryTableProcessor()
    res = processor._strip_table_content(str12)
    assert res == "balahbalah ..."

    str3 = "    balahbalah \n balahbalah "
    res = processor._strip_table_content(str3)
    assert res == "balahbalah \n balahbalah"

    str4 = "Here is the table in json format.\n```json balahbalahbalahbalah ```"
    res = processor._strip_table_content(str4)
    assert res == "balahbalahbalahbalah"

    str5 = "balahbalahbalahbalah \n balahbalahbalahbalah"
    res = processor._strip_table_content(str5)
    assert res == "balahbalahbalahbalah \n balahbalahbalahbalah"



