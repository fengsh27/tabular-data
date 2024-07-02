
import pytest
from extractor.generated_table_processor import GeneratedPKSummaryTableProcessor
from io import StringIO
import pandas as pd

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