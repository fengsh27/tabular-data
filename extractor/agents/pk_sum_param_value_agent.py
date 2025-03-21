
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_sum_common_agent import PKSumCommonAgentResult, RetryException

PARAMETER_VALUE_PROMPT = ChatPromptTemplate.from_template("""
The following main table contains pharmacokinetics (PK) data:  
{processed_md_table_aligned}
Here is the table caption:  
{caption}
From the main table above, I have extracted the following columns to create Subtable 1:  
{extracted_param_types}  
Below is Subtable 1:
{processed_md_table_aligned_with_1_param_type_and_value}
Please review the information in Subtable 1 row by row and complete Subtable 2.
Subtable 2 should have the following column headers only:  

**Main value, Statistics type, Variation type, Variation value, Interval type, Lower bound, Upper bound, P value** 

Main value: the value of main parameter (not a range). 
Statistics type: the statistics method to summary the Main value, like 'Mean,' 'Median,' 'Count,' etc. **This column is required and must be filled in.**
Variation type: the variability measure (describes how spread out the data is) associated with the Main value, like 'Standard Deviation (SD),' 'CV%,' etc.
Variation value: the value (not a range) that corresponds to the specific variation.
Interval type: the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like '95% CI,' 'Range,' 'IQR,' etc.
Lower bound: the lower bound value of the interval.
Upper bound: is the upper bound value of the interval.
P value: P-value.

Please Note:
(1) An interval consisting of two numbers must be placed separately into the Low limit and High limit fields; it is prohibited to place it in the Variation value field.
(2) For values that do not need to be filled, enter "N/A".
(3) Strictly ensure that you process only rows 0 to {md_table_aligned_with_1_param_type_and_value_max_row_index} from the Subtable 1 (which has {md_table_aligned_with_1_param_type_and_value_rows} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) For rows in Subtable 1 that can not be extracted, enter "N/A" for the entire row.
(3) **Important:** Please return Subtable 2 as a list of lists, excluding the headers. Ensure all values are converted to strings.
(4) **Absolutely no calculations are allowed—every value must be taken directly from Subtable 1 without any modifications.**  
(5) The final list should be like this:
[["0.162", "Mean", "SD", "0.090", "N/A", "N/A", "N/A", ".67"], ["0.428", "Mean", "SD", "0.162", "N/A", "N/A", "N/A", ".015"]]
""")

def get_parameter_value_prompt(
    md_table_aligned: str,
    md_table_aligned_with_1_param_type_and_value: str,
    caption: str,
):
    # Extract the first line (headers) from the provided subtable
    first_line = md_table_aligned_with_1_param_type_and_value.strip().split("\n")[0]
    headers = [col.strip() for col in first_line.split("|") if col.strip()]
    extracted_param_types = f""" "{'", "'.join(headers)}" """
    rows_num = markdown_to_dataframe(md_table_aligned_with_1_param_type_and_value)
    return PARAMETER_VALUE_PROMPT.format(
        processed_md_table_aligned=display_md_table(md_table_aligned),
        caption=caption,
        extracted_param_types=extracted_param_types,
        processed_md_table_aligned_with_1_param_type_and_value=display_md_table(
            md_table_aligned_with_1_param_type_and_value
        ),
        md_table_aligned_with_1_param_type_and_value_max_row_index=rows_num-1,
        md_table_aligned_with_1_param_type_and_value_rows=rows_num,
    )

class ParameterValueResult(PKSumCommonAgentResult):
    """ Parameter Value Extraction Result """
    extracted_param_values: List[List[float]] = Field(description="""a list of lists containing parameter values, like 
[["0.162", "Mean", "SD", "0.090", "N/A", "N/A", "N/A", ".67"], ["0.428", "Mean", "SD", "0.162", "N/A", "N/A", "N/A", ".015"]]""")

def post_process_matched_list(
    res: ParameterValueResult,
    expected_rows: int,
) -> str:
    matched_values = res.extracted_param_values

    # validation
    if not matched_values:
        raise ValueError("Parameter value extraction failed: No valid values found.")

    df_table = pd.DataFrame(matched_values, columns=[
        'Main value', 'Statistics type', 'Variation type', 'Variation value',
        'Interval type', 'Lower bound', 'Upper bound', 'P value'
    ])    
    if df_table.shape[0] != expected_rows:
        raise RetryException(
            "Wrong answer example:\n" + str(res.extracted_param_values) + f"\nWhy it's wrong:\nMismatch: Expected {expected_rows} rows, but got {df_table.shape[0]} extracted values."
        )
    return dataframe_to_markdown(df_table)


