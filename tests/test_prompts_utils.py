from extractor.prompts_utils import (
    TableExtractionPromptsGenerator,
)
from extractor.constants import PROMPTS_NAME_PK, PROMPTS_NAME_PE


def test_TableExtractionPKSummaryPromptsGenerator():
    generator = TableExtractionPromptsGenerator(PROMPTS_NAME_PK)
    prmpts = generator.generate_system_prompts()
    columns = prmpts[176:223]
    assert columns == "DN,Ana,Sp,Pop,PS,SN,PT,V,U,SS,VT,VV,IT,LL,HL,PV"


def test_TableExtractionPESummaryPromptsGenerator():
    generator = TableExtractionPromptsGenerator(PROMPTS_NAME_PE)
    prmpts = generator.generate_system_prompts()
    columns = prmpts[176:210]
    assert columns == "C/R,EX,OU,ST,V,U,VS,VV,IT,IL,IH,PV"
