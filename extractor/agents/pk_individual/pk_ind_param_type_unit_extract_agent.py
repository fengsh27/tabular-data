from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgentResult

logger = logging.getLogger(__name__)

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
(1) Refer to the "{key_with_parameter_type}" column in Subtable 1 to construct three separate lists: one for a new "Parameter type", 
  one for "Parameter unit", and one for "Parameter value". If the information in Subtable 1 is too coarse or ambiguous, you may need 
  to refer to the main table and its caption to refine and clarify your summarized "Parameter type" and "Parameter unit", 
  and if necessary, "Parameter value".
(1a) **No-Loss Normalization Rule for "Parameter type":**
    - **Do not drop or merge away any subcomponents.** Preserve every informative token (e.g., matrix, method, timing).
    - If "Parameter type" is a tuple/list (e.g., `('Cordocentesis', 'Serum')`), **concatenate all elements in order** using `" - "` as the delimiter.  
      Examples:  
      - `('Cordocentesis', 'Serum')` → `Cordocentesis-Serum`  
      - `('Maternal', 'Plasma', 'Trough')` → `Maternal-Plasma-Trough`  
      - Nested/compound values: `(('Umbilical', 'Vein'), 'Plasma')` → `Umbilical-Vein-Plasma`
    - Remove brackets/quotes only; **keep original wording and capitalization** (no abbreviations unless already present).
    - If already a single string, use it as-is (after trimming whitespace).
(2) Return a tuple containing three lists:  
    - The first list should contain the extracted "Parameter type" values.  
    - The second list should contain the corresponding "Parameter unit" values.  
    - The Third list should contain the corresponding "Parameter value" values.  
(3) **Strictly ensure that you process only rows 0 to {row_max_index} from the column "{key_with_parameter_type}".**  
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less. 
(4) For rows in Subtable 1 that can not be extracted, enter "N/A" for the entire row. 
(5) The returned list should be like this:  
    (["Parameter type 1", "Parameter type 2", ...], ["Unit 1", "Unit 2", ...], ["Value 1", "Value 2", ...])  
(6) **Keep the parameter type simple and informative (e.g. "Concentration", "Cmax", "Tmax" etc)**, **while still applying the No-Loss rule above 
  for composite labels** (e.g., keep matrices/contexts like `Cordocentesis-Serum` intact).
""")


class ExtractedParamTypeUnits(BaseModel):
    parameter_types: List[str] = Field(description="Extracted 'Parameter type' values.")
    parameter_units: List[str] = Field(
        description="Corresponding 'Parameter unit' values."
    )
    parameter_values: List[str] = Field(
        description="Corresponding 'Parameter value' values."
    )


class ParamTypeUnitExtractionResult(PKIndCommonAgentResult):
    """Unit Extraction Result"""

    extracted_param_units: ExtractedParamTypeUnits = Field(
        description="An object with lists of extracted parameter types and their corresponding units and values."
    )


def get_param_type_unit_extraction_prompt(
    md_table_aligned: str, md_sub_table: str, col_mapping: dict, caption: str
) -> str | None:
    parameter_type_count = list(col_mapping.values()).count("Parameter type")
    # parameter_unit_count = list(match_dict.values()).count("Parameter unit")
    if parameter_type_count == 1:
        key_with_parameter_type = [
            key for key, value in col_mapping.items() if value == "Parameter type"
        ][0]
        return UNIT_EXTRACTION_PROMPT.format(
            processed_md_table_aligned=display_md_table(md_table_aligned),
            caption=caption,
            processed_md_sub_table=display_md_table(md_sub_table),
            key_with_parameter_type=key_with_parameter_type,
            row_max_index=markdown_to_dataframe(md_sub_table).shape[0] - 1,
        )
    return None


# def pre_process_param_type_unit(md_table: str, col_mapping: dict):
#     parameter_type_count = list(col_mapping.values()).count("Parameter type")
#     parameter_unit_count = list(col_mapping.values()).count("Parameter unit")
#     if parameter_type_count == 1 and parameter_unit_count == 0:
#         return True
#     return False


def post_process_validate_matched_tuple(
    res: ParamTypeUnitExtractionResult,
    md_table: str,
    col_mapping: dict,
):
    matched_tuple = (
        res.extracted_param_units.parameter_types,
        res.extracted_param_units.parameter_units,
        res.extracted_param_units.parameter_values,
    )
    expected_rows = markdown_to_dataframe(md_table).shape[0]
    if len(matched_tuple[0]) != expected_rows or len(matched_tuple[1]) != expected_rows or len(matched_tuple[2]) != expected_rows:
        error_msg = (
            f"Mismatch: Expected {expected_rows} rows, but got {len(matched_tuple[0])} (types) and {len(matched_tuple[1])} (units), and "
            f"{len(matched_tuple[2])} (values)"
        )
        logger.error(error_msg)
        raise RetryException(error_msg)

    return matched_tuple
