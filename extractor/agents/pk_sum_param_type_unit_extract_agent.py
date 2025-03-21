
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_sum_common_agent import PKSumCommonAgentResult

UNIT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
The following main table contains pharmacokinetics (PK) data:  
{processed_md_table_aligned}
Here is the table caption:  
{caption}
From the main table above, I have extracted some columns to create Subtable 1:  
Below is Subtable 1:
{processed_md_sub_table}
Please note that the column "{key_with_parameter_type}" in Subtable 1 roughly represents the parameter type.
Carefully analyze the table and follow these steps:  
(1) Refer to the "{key_with_parameter_type}" column in Subtable 1 to construct two separate lists: one for a new "Parameter type" and another for "Parameter unit". If the information in Subtable 1 is too coarse or ambiguous, you may need to refer to the main table and its caption to refine and clarify your summarized "Parameter type" and "Parameter unit".
(2) Return a tuple containing two lists:  
    - The first list should contain the extracted "Parameter type" values.  
    - The second list should contain the corresponding "Parameter unit" values.  
(3) **Strictly ensure that you process only rows 0 to {row_max_index} from the column "{key_with_parameter_type}".**  
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less. 
(4) For rows in Subtable 1 that can not be extracted, enter "N/A" for the entire row. 
(5) The returned list should be like this:  
    (["Parameter type 1", "Parameter type 2", ...], ["Unit 1", "Unit 2", ...])>>  
""")

class ExtractedParamTypeUnits(BaseModel):
    parameter_types: List[str] = Field(description="Extracted 'Parameter type' values.")
    parameter_units: List[str] = Field(description="Corresponding 'Parameter unit' values.")

class ParamTypeUnitExtractionResult(PKSumCommonAgentResult):
    """ Unit Extraction Result """
    extracted_param_units: ExtractedParamTypeUnits = Field(description="An object with lists of extracted parameter types and their corresponding units.")

def get_param_type_unit_extraction_prompt(
    md_table_aligned: str, 
    md_sub_table: str, 
    col_mapping: dict,
    caption: str
) -> str | None:
    parameter_type_count = list(col_mapping.values()).count("Parameter type")
    # parameter_unit_count = list(match_dict.values()).count("Parameter unit")
    if parameter_type_count == 1:
        key_with_parameter_type = [key for key, value in col_mapping.items() if value == "Parameter type"][0]
        return UNIT_EXTRACTION_PROMPT.format(
            processed_md_table_aligned=display_md_table(md_table_aligned),
            caption=caption,
            processed_md_sub_table=display_md_table(md_sub_table),
            key_with_parameter_type=key_with_parameter_type,
            row_max_index=markdown_to_dataframe(md_sub_table).shape[0] - 1
        )
    return None

def pre_process_param_type_unit(md_table: str, col_mapping: dict):
    parameter_type_count = list(col_mapping.values()).count("Parameter type")
    parameter_unit_count = list(col_mapping.values()).count("Parameter unit")
    if parameter_type_count == 1 and parameter_unit_count == 0:
        return True
    return False
    
def post_process_validate_matched_tuple(
    res: ParamTypeUnitExtractionResult,
    md_table: str,
    col_mapping: dict,
):
    matched_tuple = (res.extracted_param_units.parameter_types, res.extracted_param_units.parameter_units)
    expected_rows = markdown_to_dataframe(md_table).shape[0]
    if len(matched_tuple[0]) != expected_rows or len(matched_tuple[1]) != expected_rows:
        raise ValueError(
            f"Mismatch: Expected {expected_rows} rows, but got {len(matched_tuple[0])} (types) and {len(matched_tuple[1])} (units).", f"\n{content}", f"\n<<{usage}>>"
        )

    return matched_tuple


