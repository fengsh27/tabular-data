
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_sum_common_agent import PKSumCommonAgentResult, RetryException

MATCHING_DRUG_PROMPT = ChatPromptTemplate.from_template("""
The following main table contains pharmacokinetics (PK) data:  
{processed_md_table_aligned}
Here is the table caption:  
{caption}
From the main table above, I have extracted the following columns to create Subtable 1:  
{extracted_param_types}  
Below is Subtable 1:
{processed_md_table_aligned_with_1_param_type_and_value}
Additionally, I have compiled Subtable 2, where each row represents a unique combination of "Drug name" - "Analyte" - "Specimen," as follows:
{processed_drug_md_table}
Carefully analyze the tables and follow these steps:  
(1) For each row in Subtable 1, find **the best matching one** row in Subtable 2. Return a list of unique row indices (as integers) from Subtable 2 that correspond to each row in Subtable 1.  
(2) **Strictly ensure that you process only rows 0 to {max_md_table_aligned_with_1_param_type_and_value_row_index} from the Subtable 1.**  
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(3) The final list should be like this:  
   [1,1,2,2,3,3]
""")

def get_matching_drug_prompt(
    md_table_aligned: str,
    md_table_aligned_with_1_param_type_and_value: str,
    drug_md_table: str,
    caption: str,
):
    first_line = md_table_aligned_with_1_param_type_and_value.strip().split("\n")[0]
    headers = [col.strip() for col in first_line.split("|") if col.strip()]
    extracted_param_types = f""" "{'", "'.join(headers)}" """
    return MATCHING_DRUG_PROMPT.format(
        processed_md_table_aligned=display_md_table(md_table_aligned),
        caption=caption,
        extracted_param_types=extracted_param_types,
        processed_md_table_aligned_with_1_param_type_and_value=display_md_table(
            md_table_aligned_with_1_param_type_and_value
        ),
        processed_drug_md_table=display_md_table(drug_md_table),
        max_md_table_aligned_with_1_param_type_and_value_row_index=markdown_to_dataframe(
            md_table_aligned_with_1_param_type_and_value
        ).shape[0] - 1,
    )

class MatchedDrugResult(PKSumCommonAgentResult):
    """ Matched Drug Result """
    matched_row_indices: List[int] = Field(description="a list of matched row indices")


def post_process_validate_matched_rows(
    res: MatchedDrugResult,
    md_table: str,
):
    match_list = res.matched_row_indices
    expected_rows = markdown_to_dataframe(md_table).shape[0]
    if len(match_list) != expected_rows:
        raise RetryException(
            f"Mismatch: Expected {expected_rows} rows, but got {len(match_list)} extracted matches."
        )
    return res

