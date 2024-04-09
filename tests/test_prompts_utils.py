import pytest

from extractor.prompts_utils import TableExtractionPromptsGenerator

def test_TableExtractionPromptsGenerator():
    generator = TableExtractionPromptsGenerator()
    prmpts = generator.generate_system_prompts()
    assert prmpts is not None