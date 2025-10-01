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

MATCHING_DRUG_PROMPT = from_system_template("""
Analyze the following pharmacokinetics (PK) data tables and perform the specified matching operation:

MAIN TABLE (PK Data):
{processed_md_table_aligned}

Caption: {caption}

SUBTABLE 1 (Extracted from Main Table):
{processed_md_table_aligned_with_1_param_type_and_value}

SUBTABLE 2 (Drug-Analyte-Specimen Combinations):
{processed_drug_md_table}

---

### **TASK:**
1. For each of rows 0-{max_md_table_aligned_with_1_param_type_and_value_row_index} in Subtable 1, find the BEST matching row in Subtable 2 based on:
   - For each row in Subtable 1, find **the best matching one** row in Subtable 2
   - Context from the table caption about the drug (e.g. lorazepam)

2. Processing Rules:
   - Only process rows 0-{max_md_table_aligned_with_1_param_type_and_value_row_index} from Subtable 1 (exactly {md_table_aligned_with_1_param_type_and_value_row_num} rows total)
   - Return indices of matching Subtable 2 rows as a Python list of integers
   - If no clear best match is identified for a given row, default to using -1. Important: This default should only be applied when no legitimate match exists after thorough evaluation of all available data.
   - Example output format: [0, 1, 2, 3, 4, 5, 6, ...]

### ** Important Instructions:**
   - You **must follow** the following steps to match the row in Subtable 1 to the row in Subtable 2:
     For each row in Subtable 1, 
      * First find the corresponding row in **main table** for the row in Subtable 1 according to row index (row index in main table is the same as the row index in Subtable 1), 
      * The row in main table provide more context,then find the best matching row in **Subtable 2** according to the row in main table.
   - As SUBTABLE 1 is extracted from MAIN TABLE in row order, if you cannot determine the best matching row in Subtable 2 for a given row in Subtable 1, 
     you can infer the best matching by referring to the row before it or after it.
     For example, main table is like this:
     | Drug Name      | Patient ID      | Parameter Type | Parameter Value |
     | -------------- | --------------- | --------------- | --------------- |
     | B1             | 1               | urine           | -               |
     | B1             | 2               | urine           | -               |
     | B1             | 3               | urine           | -               |
     | B2             | 1               | urine           | -               |
     | B2             | 2               | urine           | -               |
     | B2             | 3               | urine           | -               |
     | B3             | 1               | urine           | -               |
     | B3             | 2               | urine           | -               |
     | B3             | 3               | urine           | -               |
     then Subtable 1 is like this:
     | Patient ID      | Parameter Type | Parameter Value |
     | -------------- | --------------- | --------------- |
     | 1               | urine           | -               |
     | 2               | urine           | -               |
     | 3               | urine           | -               |
     | 1               | urine           | -               |
     | 2               | urine           | -               |
     | 3               | urine           | -               |
     | 1               | urine           | -               |
     | 2               | urine           | -               |
     | 3               | urine           | -             |
     then Subtable 2 is like this:
     | Drug Name       | Analyte       | Specimen       |
     | -------------- | --------------- | --------------- |
     | B1             | B1              | urine           |
     | B2             | B2              | urine           |
     | B3             | B3              | urine           |
     
     1. For the row 0, 1 and 2 in Subtable 1, the best matching row in Subtable 2 is 0 (index).
     As Subtable 1 is extracted from main table in row order, the corresponding rows in main table for row 0, 1 and 2 in Subtable 1 are 0, 1 and 2 (index).
     Then, we can determin their drug name from main table are B1, so the best matching rows in Subtable 2 for row 0, 1 and 2 in main table are [0, 0, 0] (index)
     Thus, the best matching rows in Subtable 2 for the row 0, 1 and 2 in Subtable 1 are [0, 0, 0] (index).
     2. For the row 3, 4 and 5 in Subtable 1, the best matching row in Subtable 2 is 1 (index).
     Likewise, as the Subtable 1 is extracted from main table in row order, the corresponding rows in main table for the row 3, 4 and 5 are 3, 4 and 5 (index).
     Thus, we can determine their drug name from main table are B2, so the best matching rows in Subtable 2 for the row 3, 4 and 5 in main table are [1, 1, 1] (index).
     3. For the row 6, 7 and 8 in Subtable 1, the best matching row in Subtable 2 is 2 (index).
     Likewise, as Subtable 1 is extracted from main table in row order, we can determine the best matching rows in main table for the row 6, 7 and 8 in Subtable 1 are 6, 7 and 8 (index).
     Thus, we can determine their drug name from main table are B3, so the best matching rows in Subtable 2 for the row 6, 7 and 8 in main table are [2, 2, 2] (index).

     So, we get the best matching rows in Subtable 2 for the row 0, 1, 2, 3, 4, 5, 6, 7 and 8 in Subtable 1 are [0, 0, 0, 1, 1, 1, 2, 2, 2] (index).



""")

# MATCHING_DRUG_PROMPT = ChatPromptTemplate.from_template("""
# The following main table contains pharmacokinetics (PK) data:
# {processed_md_table_aligned}
# Here is the table caption:
# {caption}
# From the main table above, I have extracted the following columns to create Subtable 1:
# {extracted_param_types}
# Below is Subtable 1:
# {processed_md_table_aligned_with_1_param_type_and_value}
# Additionally, I have compiled Subtable 2, where each row represents a unique combination of "Drug name" - "Analyte" - "Specimen," as follows:
# {processed_drug_md_table}
# Carefully analyze the tables and follow these steps:
# (1) For each row in Subtable 1, find **the best matching one** row in Subtable 2. Return a list of unique row indices (as integers) from Subtable 2 that correspond to each row in Subtable 1.
# (2) **Strictly ensure that you process only rows 0 to {max_md_table_aligned_with_1_param_type_and_value_row_index} from the Subtable 1.**
# - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.
# (3) The final list should be like this:
# [1,1,2,2,3,3]
# """)


def get_matching_drug_prompt(
    md_table_aligned: str,
    md_table_aligned_with_1_param_type_and_value: str,
    drug_md_table: str,
    caption: str,
):
    first_line = md_table_aligned_with_1_param_type_and_value.strip().split("\n")[0]
    headers = [col.strip() for col in first_line.split("|") if col.strip()]
    extracted_param_types = f""" "{'", "'.join(headers)}" """
    df_aligned_with_1_param_type_and_value = markdown_to_dataframe(
        md_table_aligned_with_1_param_type_and_value
    )
    return MATCHING_DRUG_PROMPT.format(
        processed_md_table_aligned=display_md_table(md_table_aligned),
        caption=caption,
        # extracted_param_types=extracted_param_types,
        processed_md_table_aligned_with_1_param_type_and_value=display_md_table(
            md_table_aligned_with_1_param_type_and_value
        ),
        processed_drug_md_table=display_md_table(drug_md_table),
        max_md_table_aligned_with_1_param_type_and_value_row_index=df_aligned_with_1_param_type_and_value.shape[
            0
        ]
        - 1,
        md_table_aligned_with_1_param_type_and_value_row_num=df_aligned_with_1_param_type_and_value.shape[
            0
        ],
    )


class MatchedDrugResult(PKIndCommonAgentResult):
    """Matched Drug Result"""

    matched_row_indices: List[int] = Field(description="a list of matched row indices")


def post_process_validate_matched_rows(
    res: MatchedDrugResult,
    md_table1: str,
    md_table2: str,
):
    match_list = res.matched_row_indices
    expected_rows = markdown_to_dataframe(md_table1).shape[0]

    matched_row_max_index = markdown_to_dataframe(md_table2).shape[0] - 1
    if len(match_list) != expected_rows:
        raise RetryException(f"""
The provided answer `{match_list}` appears incorrect because:  
- **Requirement**: Expected {expected_rows} matching indices (one for each row in Subtable 1).  
- **Issue**: Only {len(match_list)} indices were returned.  

Please ensure future responses:  
1. Match **all {expected_rows} rows** from Subtable 1 to Subtable 2.  
2. Carefully validate the output format before submission.  
""")
        # raise RetryException(
        #     f"You Wrong answer: {match_list}. Why it's wrong, mismatch: Expected {expected_rows} rows, but got {len(match_list)} extracted matches."
        # )
    for i in range(len(match_list)):
        ix = match_list[i]
        if ix == -1:
            logger.error(
                f"drug matching: the {i}th row can't identify a best match in Subtable2, fill it with 0."
            )
            match_list[i] = 0
            ix = 0
        if ix < 0 or ix > matched_row_max_index:
            error_msg = f"Wrong answer: {match_list}. Why it's wrong, row index should between 0 to {matched_row_max_index}. {ix} is not correct."
            logger.error(error_msg)
            raise RetryException(error_msg)
    return match_list
