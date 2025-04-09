import json
import pytest
from extractor.generated_table_processor import (
    GeneratedPKSummaryTableProcessor,
    JsonEnclosePropertyNameInQuotesPlugin,
)
from io import StringIO
import pandas as pd

from extractor.utils import convert_csv_table_to_dataframe
from extractor.constants import (
    PKSUMMARY_TABLE_OUTPUT_COLUMNS,
    PROMPTS_NAME_PE,
    PROMPTS_NAME_PK,
)

# 30950674
json_content1 = """
[{"DN":"","Ana":"Actual body weight","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":6.2,"U":"kg","SS":"","VV":"","VT":"","IT":"IQR","LL":3.6,"HL":7.4,"PV":""}, {"DN":"","Ana":"Actual body weight","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"kg","SS":"","VV":"","VT":"","IT":"Range","LL":0.8,"HL":10.5,"PV":""}, {"DN":"","Ana":"Height","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":64,"U":"cm","SS":"","VV":"","VT":"","IT":"IQR","LL":56,"HL":71,"PV":""}, {"DN":"","Ana":"Height","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"cm","SS":"","VV":"","VT":"","IT":"Range","LL":38,"HL":76,"PV":""}, {"DN":"","Ana":"Body surface area","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":0.33,"U":"m²","SS":"","VV":"","VT":"","IT":"IQR","LL":0.24,"HL":0.38,"PV":""}, {"DN":"","Ana":"Body surface area","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"m²","SS":"","VV":"","VT":"","IT":"Range","LL":0.09,"HL":0.47,"PV":""}, {"DN":"","Ana":"Gestational age","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":39,"U":"weeks","SS":"","VV":"","VT":"","IT":"IQR","LL":31,"HL":40,"PV":""}, {"DN":"","Ana":"Gestational age","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"weeks","SS":"","VV":"","VT":"","IT":"Range","LL":25,"HL":42,"PV":""}, {"DN":"","Ana":"Postnatal age","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":157,"U":"days","SS":"","VV":"","VT":"","IT":"IQR","LL":112,"HL":238,"PV":""}, {"DN":"","Ana":"Postnatal age","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"days","SS":"","VV":"","VT":"","IT":"Range","LL":31,"HL":357,"PV":""}, {"DN":"","Ana":"Postmenstrual age","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":58,"U":"weeks","SS":"","VV":"","VT":"","IT":"IQR","LL":46,"HL":72,"PV":""}, {"DN":"","Ana":"Postmenstrual age","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"weeks","SS":"","VV":"","VT":"","IT":"Range","LL":31,"HL":91,"PV":""}, {"DN":"","Ana":"eGFR","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":21.24,"U":"mL/min","SS":"","VV":"","VT":"","IT":"IQR","LL":14.1,"HL":27.66,"PV":""}, {"DN":"","Ana":"eGFR","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"mL/min","SS":"","VV":"","VT":"","IT":"Range","LL":1.26,"HL":53.7,"PV":""}, {"DN":"","Ana":"Vd","Sp":"","Pop":"","PS":"","SN":"","PT":"Mean","V":3.62,"U":"L","SS":"","VV":1.94,"VT":"SD","IT":"","LL":"","HL":"","PV":""}, {"DN":"","Ana":"Vd","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":3.5,"U":"L","SS":"","VV":"","VT":"","IT":"IQR","LL":2.64,"HL":4.35,"PV":""}, {"DN":"","Ana":"Vd","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"L","SS":"","VV":"","VT":"","IT":"Range","LL":0.31,"HL":8.88,"PV":""}, {"DN":"","Ana":"Vd (L/kg)","Sp":"","Pop":"","PS":"","SN":"","PT":"Mean","V":0.7,"U":"L/kg","SS":"","VV":0.48,"VT":"SD","IT":"","LL":"","HL":"","PV":""}, {"DN":"","Ana":"Vd (L/kg)","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":0.5,"U":"L/kg","SS":"","VV":"","VT":"","IT":"IQR","LL":0.39,"HL":0.94,"PV":""}, {"DN":"","Ana":"Vd (L/kg)","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"L/kg","SS":"","VV":"","VT":"","IT":"Range","LL":0.28,"HL":2.31,"PV":""}, {"DN":"","Ana":"CL","Sp":"","Pop":"","PS":"","SN":"","PT":"Mean","V":0.749,"U":"L/h","SS":"","VV":0.454,"VT":"SD","IT":"","LL":"","HL":"","PV":""}, {"DN":"","Ana":"CL","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":0.737,"U":"L/h","SS":"","VV":"","VT":"","IT":"IQR","LL":0.414,"HL":0.955,"PV":""}, {"DN":"","Ana":"CL","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"L/h","SS":"","VV":"","VT":"","IT":"Range","LL":0.02,"HL":1.82,"PV":""}, {"DN":"","Ana":"CL (L/h/kg)","Sp":"","Pop":"","PS":"","SN":"","PT":"Mean","V":0.124,"U":"L/h/kg","SS":"","VV":0.052,"VT":"SD","IT":"","LL":"","HL":"","PV":""}, {"DN":"","Ana":"CL (L/h/kg)","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":0.112,"U":"L/h/kg","SS":"","VV":"","VT":"","IT":"IQR","LL":0.095,"HL":0.133,"PV":""}, {"DN":"","Ana":"CL (L/h/kg)","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"L/h/kg","SS":"","VV":"","VT":"","IT":"Range","LL":0.025,"HL":0.268,"PV":""}, {"DN":"","Ana":"t½","Sp":"","Pop":"","PS":"","SN":"","PT":"Mean","V":4.3,"U":"h","SS":"","VV":2.6,"VT":"SD","IT":"","LL":"","HL":"","PV":""}, {"DN":"","Ana":"t½","Sp":"","Pop":"","PS":"","SN":"","PT":"Median","V":3.8,"U":"h","SS":"","VV":"","VT":"","IT":"IQR","LL":2.5,"HL":5.4,"PV":""}, {"DN":"","Ana":"t½","Sp":"","Pop":"","PS":"","SN":"","PT":"Min–Max","V":"","U":"h","SS":"","VV":"","VT":"","IT":"Range","LL":1.5,"HL":11.3,"PV":""}, {"DN":"","Ana":"Actual body weight","Sp":"","Pop":"","PS":"","SN":"","PT":"r²","V":0.6771,"U":"","SS":"","VV":"","VT":"","IT":"95% CI","LL":0.0958,"HL":0.1867,"PV":"<.0001"}, {"DN":"","Ana":"Height","Sp":"","Pop":"","PS":"","SN":"","PT":"r²","V":0.5447,"U":"","SS":"","VV":"","VT":"","IT":"95% CI","LL":0.0182,"HL":0.0453,"PV":"<.0001"}, {"DN":"","Ana":"Body surface area","Sp":"","Pop":"","PS":"","SN":"","PT":"r²","V":0.662,"U":"","SS":"","VV":"","VT":"","IT":"95% CI","LL":2.38,"HL":4.759,"PV":"<.0001"}, {"DN":"","Ana":"Gestational age","Sp":"","Pop":"","PS":"","SN":"","PT":"r²","V":0.3822,"U":"","SS":"","VV":"","VT":"","IT":"95% CI","LL":0.0192,"HL":0.0753,"PV":".0022"}, {"DN":"","Ana":"Postnatal age","Sp":"","Pop":"","PS":"","SN":"","PT":"r²","V":0.5168,"U":"","SS":"","VV":"","VT":"","IT":"95% CI","LL":0.0125,"HL":0.033,"PV":".0002"}, {"DN":"","Ana":"Postmenstrual age","Sp":"","Pop":"","PS":"","SN":"","PT":"r²","V":0.6036,"U":"","SS":"","VV":"","VT":"","IT":"95% CI","LL":0.0122,"HL":0.0271,"PV":"<.0001"}, {"DN":"","Ana":"eGFR","Sp":"","Pop":"","PS":"","SN":"","PT":"r²","V":0.5271,"U":"","SS":"","VV":"","VT":"","IT":"95% CI","LL":0.9294,"HL":2.401,"PV":".0001"}]
"""


@pytest.mark.skip(
    "csv parser can't handle comma in value correctly (have tried many ways)"
)
def test_pandas_read_csv():
    csv_header = ", ".join(PKSUMMARY_TABLE_OUTPUT_COLUMNS)
    csv_value1 = "lorazepam, Lorazepam enantiomeric mixture, blood, maternal, N/A, 8, concentration, 9.91, ng/ml, mean, None, None, 95% CI, 7.68, 12.14, None"
    csv_value2 = "lorazepam, Lorazepam enantiomeric mixture, `cord blood,blood`, `neonate,maternal`, N/A, 8, cord blood/maternal blood, 0.73, None, mean, None, None, 95% CI, 0.52, 0.94, None"
    csv_str = f"{csv_header}\n{csv_value1}\n{csv_value2}\n"
    # buf = StringIO(csv_str)
    df = convert_csv_table_to_dataframe(csv_str)
    assert df is not None
    for i in range(16):
        the_val = df.iloc[0, i]
        print(the_val)


@pytest.mark.parametrize(
    "pmid, expected",
    [
        ("29943508", (6, 16)),
        ("34183327", (30, 16)),
        ("30950674_gemini", (30, 16)),
        ("34114632", (24, 16)),
        ("24132975_pe", (44, 12)),
    ],
)
def test_converter(pmid, expected):
    with open(f"./tests/data/{pmid}_result.json", "r") as fobj:
        res_str = fobj.read()
        prompts_type = PROMPTS_NAME_PE if "_pe" in pmid else PROMPTS_NAME_PK
        processor = GeneratedPKSummaryTableProcessor(prompts_type=prompts_type)
        csv_str = processor.process_content(res_str)
        buf = StringIO(csv_str)
        df = pd.read_csv(buf)
        assert df is not None
        assert tuple(df.shape) == expected


def test_16143486_gemini_result_json():
    with open("./tests/data/16143486_gemini_result.json", "r") as fobj:
        res_str = fobj.read()
        processor = GeneratedPKSummaryTableProcessor()
        csv_str = processor.process_content(res_str)
        buf = StringIO(csv_str)
        df = pd.read_csv(buf)
        assert df is not None
        assert tuple(df.shape) == (29, 16)


def test_16143486_gemini_result_json_1():
    with open("./tests/data/16143486_gemini_result_1.json", "r") as fobj:
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


def test_36396314_gpt_4o_error_result_json_1():
    with open("./tests/data/36396314_gpt_4o_error_result.txt", "r") as fobj:
        res_str = fobj.read()
        processor = GeneratedPKSummaryTableProcessor()
        try:
            csv_str = processor.process_content(res_str)
        except json.JSONDecodeError as e:
            # expected error
            print(str(e))
            return
        buf = StringIO(csv_str)
        df = pd.read_csv(buf)
        assert df is not None
        assert tuple(df.shape) == (30, 16)


def test_plugin():
    p = JsonEnclosePropertyNameInQuotesPlugin(PROMPTS_NAME_PK)
    processed_str = p.process(json_content1)
    assert processed_str is not None


def test_convert_json_to_csv():
    prcr = GeneratedPKSummaryTableProcessor()
    csv_content = prcr.process_content(json_content1)
    assert csv_content is not None
    with open("./temp1.csv", "w") as fobj:
        fobj.write(csv_content)
