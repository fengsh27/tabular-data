
import pytest

from extractor.agents.pk_individual.pk_ind_param_type_unit_extract_agent import (
    try_fix_error_param_type_unit,
    ParamTypeUnitExtractionResult,
    ExtractedParamTypeUnits,
)

res = ParamTypeUnitExtractionResult(
    reasoning_process="",
    extracted_param_units=ExtractedParamTypeUnits(
        parameter_types=['Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'N/A', 'N/A', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'N/A', 'N/A', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'N/A', 'N/A', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'N/A', 'N/A', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL', 'Postpartum  12 Weeks-Plasma  Level,  ng/mL'],
        parameter_units=['ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'N/A', 'N/A', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'N/A', 'N/A', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'ng/mL', 'N/A', 'N/A', 'ng/mL', 'ng/mL', 'ng/mL'], 
        parameter_values=['1', '18', '36', '2', '27', '75', '3b', 'N/A', 'N/A', '4', 'NA', '5', '63', '6', '21', '50', '7', 'N/A', 'N/A', '8', '21', '29', '9', 'N/A', 'N/A', '10', 'N/A', 'N/A', '11', '99', '183']
    ),
)
md_table = "| ('Unnamed: 0_level_0', 'Subject') | Parameter type | Parameter value |\n| --- | --- | --- |\n| 1 | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 1 |\n| S-citalopram | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 18 |\n| R-citalopram | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 36 |\n| 2 | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 2 |\n| S-citalopram | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 27 |\n| R-citalopram | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 75 |\n| 3b | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 3b |\n| S-citalopram | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | … |\n| R-citalopram | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | … |\n| 4 | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 4 |\n| S-citalopram | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| 5 | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 5 |\n| S-citalopram | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 63 |\n| 6 | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 6 |\n| S-sertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 21 |\n| N-desmethylsertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 50 |\n| 7 | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 7 |\n| S-sertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| N-desmethylsertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| 8 | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 8 |\n| S-sertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 21 |\n| N-desmethylsertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 29 |\n| 9 | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 9 |\n| S-sertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| N-desmethylsertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| 10 | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 10 |\n| S-sertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| N-desmethylsertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| 11 | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 11 |\n| S-sertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 99 |\n| N-desmethylsertraline | ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 183 |"
col_mapping = {"('Unnamed: 0_level_0', 'Subject')": 'Patient ID', "('20 Weeks', 'Dose,  mg/d')": 'Uncategorized', "('20 Weeks', 'Plasma  Level,  ng/mL')": 'Parameter value', "('30 Weeks', 'Dose,  mg/d')": 'Uncategorized', "('30 Weeks', 'Plasma  Level,  ng/mL')": 'Parameter value', "('36 Weeks', 'Dose,  mg/d')": 'Uncategorized', "('36 Weeks', 'Plasma  Level,  ng/mL')": 'Parameter value', "('Delivery', 'Dose,  mg/d')": 'Uncategorized', "('Delivery', 'Plasma  Level,  ng/mL')": 'Parameter value', "('Postpartum 2 Weeks', 'Dose,  mg/d')": 'Uncategorized', "('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL')": 'Parameter value', "('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d')": 'Uncategorized', "('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL')": 'Parameter value', "('Postpartum  12 Weeks', 'Dose,  mg/d')": 'Uncategorized', "('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL')": 'Parameter value', 'Parameter type': 'Parameter type'}

def test_try_fix_error_param_type_unit():
    type, unit, values = try_fix_error_param_type_unit(res, md_table, col_mapping)

    assert len(type) == len(unit)
    assert len(unit) == len(values)
    assert len(type) == 31



