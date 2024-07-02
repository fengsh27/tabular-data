import pytest
from io import StringIO
import pandas as pd

from extractor.prompts_utils import (
    TableExtractionPKSummaryPromptsGenerator,
)

def test_TableExtractionPKSummaryPromptsGenerator():
    generator = TableExtractionPKSummaryPromptsGenerator()
    prmpts = generator.generate_system_prompts("PK Prompts")
    assert prmpts is not None



