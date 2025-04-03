
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table, from_system_template
from extractor.agents.pk_summary.pk_sum_common_agent import (
    PKSumCommonAgentResult, 
    RetryException,
)


MATCHING_PATIENT_PROMPT=from_system_template("""
The following main table contains pharmacokinetics (PK) data:  
{processed_md_table_aligned}
Here is the table caption:  
{caption}
From the main table above, I have extracted the following columns to create Subtable 1:  
{extracted_param_types}  
Below is Subtable 1:
{processed_md_table_aligned_with_1_param_type_and_value}
Additionally, I have compiled Subtable 2, where each row represents a unique combination of "Population" - "Pregnancy stage" - "Subject N," as follows:
{processed_patient_md_table}
Carefully analyze the tables and follow these steps:  
(1) For each row in Subtable 1, find **the best matching one** row in Subtable 2. Return a list of unique row indices (as integers) from Subtable 2 that correspond to each row in Subtable 1.  
(2) **Strictly ensure that you process only rows 0 to {max_md_table_aligned_with_1_param_type_and_value_row_index} from the Subtable 1.**  
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(3) The "Subject N" values within each population group sometimes differ slightly across parameters. This reflects data availability for each specific parameter within that age group. 
    - For instance, if the total N is 10 but a specific data point corresponds to 9, the correct Subject N for that row should be 9. It is essential to ensure that each row is matched with the appropriate Subject N accordingly.
(4) The final list should be like this without removing duplicates or sorting:
    [1,1,2,2,3,3]
(5) In rare cases where a row in Subtable 1 cannot be matched, return -1 for that row. This should only be used when absolutely necessary.
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
    return MATCHING_PATIENT_PROMPT.format(
        processed_md_table_aligned=display_md_table(md_table_aligned),
        caption=caption,
        extracted_param_types=extracted_param_types,
        processed_md_table_aligned_with_1_param_type_and_value=display_md_table(
            md_table_aligned_with_1_param_type_and_value
        ),
        processed_patient_md_table=display_md_table(patient_md_table),
        max_md_table_aligned_with_1_param_type_and_value_row_index=markdown_to_dataframe(
            md_table_aligned_with_1_param_type_and_value
        ).shape[0] - 1
    )

class MatchedPatientResult(PKSumCommonAgentResult):
    """ Matched Patients Result """
    matched_row_indices: List[int] = Field(description="a list of matched row indices")

def post_process_validate_matched_patients(
    res: MatchedPatientResult,
    md_table: str,
):
    match_list = res.matched_row_indices
    expected_rows = markdown_to_dataframe(md_table).shape[0]
    if len(match_list) != expected_rows:
        raise RetryException(f"""
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
""")
    
    return match_list
