from typing import List
from pydantic import Field
import logging

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table, from_system_template
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_individual.pk_ind_common_agent import (
    PKIndCommonAgentResult,
)

logger = logging.getLogger(__name__)

MATCHING_PATIENT_SYSTEM_PROMPT = from_system_template("""
You are given a main table containing pharmacokinetics (PK) data:  
**Main Table:**  
{processed_md_table_aligned}  

**Caption:**  
{caption}
From this main table, I have extracted a subset of rows based on specific parameters to form **Subtable 1**:  
{extracted_param_types}  

**Subtable 1 (Single Parameter Type and single Parameter Value per Row):**  
{processed_md_table_aligned_with_1_param_type_and_value}  

I have also compiled **Subtable 2**, where each row corresponds to a unique combination of:  
**"Patient ID" - "Population" – "Pregnancy stage"**  
{processed_patient_md_table}  

### Task:
Carefully analyze the tables and follow these instructions step by step:

1. **Match Each Row in Subtable 1 to Subtable 2:**
   - For each row in Subtable 1 (rows 0 to {max_md_table_aligned_with_1_param_type_and_value_row_index}), find the **best matching row** in Subtable 2.
   - First, find the corresponding row in the **Main Table** using the **Parameter Value** from Subtable 1.
   - Then, use the associated **"Patient ID"** value to identify the matching row in Subtable 2.

2. **Strict Row Range:**
   - Only process rows **0 to {max_md_table_aligned_with_1_param_type_and_value_row_index}** in Subtable 1.
   - Your final output list must include exactly the same number of entries as there are rows in Subtable 1.

3. **Output Format:**
   - Return a Python-style list containing the row indices (integers) of Subtable 2 that best match each row in Subtable 1.
   - Do not sort or deduplicate the list. The output should follow the order of Subtable 1:
     ```
     [matched_index_row_0, matched_index_row_1, ..., matched_index_row_N]
     ```

5. **If No Match Found:**
   - If a row in Subtable 1 cannot be matched even after applying all criteria, return `-1` for that row.
   - Use this only as a last resort.
""")

def get_matching_patient_prompt(
    md_table_aligned: str,
    md_table_aligned_with_1_param_type_and_value: str,
    patient_md_table: str,
    caption: str,
):
    first_line = md_table_aligned_with_1_param_type_and_value.strip().split("\n")[0]
    headers = [col.strip() for col in first_line.split("|") if col.strip()]
    extracted_param_types = f""" "{'", "'.join(headers)}" """
    return MATCHING_PATIENT_SYSTEM_PROMPT.format(
        processed_md_table_aligned=display_md_table(md_table_aligned),
        caption=caption,
        extracted_param_types=extracted_param_types,
        processed_md_table_aligned_with_1_param_type_and_value=display_md_table(
            md_table_aligned_with_1_param_type_and_value
        ),
        processed_patient_md_table=display_md_table(patient_md_table),
        max_md_table_aligned_with_1_param_type_and_value_row_index=markdown_to_dataframe(
            md_table_aligned_with_1_param_type_and_value
        ).shape[0]
        - 1,
    )


class MatchedPatientResult(PKIndCommonAgentResult):
    """Matched Patients Result"""

    matched_row_indices: List[int] = Field(description="a list of matched row indices")


def post_process_validate_matched_patients(
    res: MatchedPatientResult,
    md_table: str,
):
    match_list = res.matched_row_indices
    expected_rows = markdown_to_dataframe(md_table).shape[0]
    if len(match_list) != expected_rows:
        error_msg = f"""
**Error Identification:**
Your previous answer `{match_list}` is incorrect because:
- Expected output length: {expected_rows} (to match all Subtable 1 rows)
- Provided output length: {len(match_list)}

**Required Correction:**
Please:
1. Carefully reprocess all {expected_rows} rows from Subtable 1
2. For each row, explicitly document your matching logic:
   - Which Subtable 2 row index you selected
   - The specific criteria used to determine the match
3. Ensure your final answer contains exactly {expected_rows} indices

**Key Requirements:**
- Maintain 1:1 correspondence between Subtable 1 rows and output indices
- Include clear justification for each match in reasoning process
- Verify the output length matches the input exactly
"""
        logger.error(error_msg)
        raise RetryException(error_msg)

    return match_list
