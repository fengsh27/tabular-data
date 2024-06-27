import pytest
from io import StringIO
import pandas as pd

from extractor.prompts_utils import (
    TableExtractionPKSummaryPromptsGenerator,
    GeneratedTableProcessor,
)

def test_TableExtractionPKSummaryPromptsGenerator():
    generator = TableExtractionPKSummaryPromptsGenerator()
    prmpts = generator.generate_system_prompts("PK Prompts")
    assert prmpts is not None

@pytest.mark.parametrize("pmid, expected", [
    ("29943508", (6,16)),
    ("34183327", (30, 16)),
])
def test_converter(pmid, expected):
    with open(f"./tests/{pmid}_result.json", "r") as fobj:
        res_str = fobj.read()
        processor = GeneratedTableProcessor()
        csv_str = processor.process_content(res_str)
        buf = StringIO(csv_str)
        df = pd.read_csv(buf)
        assert df is not None
        assert tuple(df.shape) == expected

