from extractor.prompts_utils import (
    TableExtractionPromptsGenerator,
)
from extractor.constants import PROMPTS_NAME_PK_SUM, PROMPTS_NAME_PK_IND, \
    PROMPTS_NAME_PK_SPEC_SUM, PROMPTS_NAME_PK_DRUG_SUM, PROMPTS_NAME_PK_POPU_SUM, PROMPTS_NAME_PE, \
    PROMPTS_NAME_PK_SPEC_IND, PROMPTS_NAME_PK_DRUG_IND, PROMPTS_NAME_PK_POPU_IND


def test_TableExtractionPKSummaryPromptsGenerator():
    generator = TableExtractionPromptsGenerator(PROMPTS_NAME_PK_SUM)
    prmpts = generator.generate_system_prompts()
    columns = prmpts[176:223]
    assert columns == "DN,Ana,Sp,Pop,PS,SN,PT,V,U,SS,VT,VV,IT,LL,HL,PV"


def test_TableExtractionPESummaryPromptsGenerator():
    generator = TableExtractionPromptsGenerator(PROMPTS_NAME_PE)
    prmpts = generator.generate_system_prompts()
    columns = prmpts[176:210]
    assert columns == "C/R,EX,OU,ST,V,U,VS,VV,IT,IL,IH,PV"
