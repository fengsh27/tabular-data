
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_sum_common_agent import PKSumCommonAgentResult, RetryException

TIME_AND_UNIT_PROMPT =  ChatPromptTemplate.from_template("""
The following table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
From the main table above, I have extracted some information to create Subtable 1:  
Below is Subtable 1:
{processed_md_table_post_processed}
Carefully analyze the table and follow these steps:  
(1) For each row in Subtable 1, add two more columns [Time value, Time unit].  
- **Time Value:** A specific moment (numerical or time range) when the row of data is recorded, or a drug dose is administered.  
  - Examples: Sampling times, dosing times, or reported observation times.  
- **Time Unit:** The unit corresponding to the recorded time point (e.g., "Hour", "Min", "Day").  
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be like this:  
`[["0-1", "Hour"], ["10", "Min"], ["N/A", "N/A"]]` (example)
(3) Strictly ensure that you process only rows 0 to {md_data_post_processed_max_row_index} from the Subtable 1 (which has {md_data_lines_after_post_process_row_num} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) Verify the source of each [Time value, Time unit] combination before including it in your answer.  
(5) **Absolutely no calculations are allowed—every value must be taken directly from the table without any modifications.** 
(6) **If no valid [Time value, Time unit] combinations are found, return the default output:**  
`[["N/A", "N/A"]]`

**Examples:**
Include:  
   - "0-12" (indicating a dosing period)  
   - "24" (indicating a time of sample collection)  
   - "5 min" (indicating a measured event)  

Do NOT include:  
   - "Tmax" (this is a pharmacokinetic parameter, NOT a recorded time)
   - "T½Beta(hr)" values (half-life parameter value, not a recorded time) 
   - "Beta(hr)" values (elimination rate constant)

""")

def get_time_and_unit_prompt(
    md_table_aligned: str, 
    md_table_post_processed: str, 
    caption: str
):
    row_num = markdown_to_dataframe(md_table_post_processed).shape[0]
    return TIME_AND_UNIT_PROMPT.format(
        processed_md_table=display_md_table(md_table_aligned),
        caption=caption,
        processed_md_table_post_processed=display_md_table(md_table_post_processed),
        md_data_post_processed_max_row_index=row_num-1,
        md_data_lines_after_post_process_row_num=row_num,
    )

class TimeAndUnitResult(PKSumCommonAgentResult):
    """ Time and Unit Extraction Result """
    times_and_units: List[List[str]] = Field(description="a list of lists, where each inner list represents [Time value, Time unit]")
    

def post_process_time_and_unit(
    res: TimeAndUnitResult,
    md_table_post_processed: str, # markdown table after post-processing
) -> str:
    match_list = res.times_and_units
    expected_rows = markdown_to_dataframe(md_table_post_processed).shape[0]
    if all(x == match_list[0] for x in match_list):
        # expand to expect_rows
        match_list = [match_list[0]] * expected_rows
    
    df_table = pd.DataFrame(match_list, columns=["Time value", "Time unit"])
    if df_table.shape[0] != expected_rows:
        raise RetryException(
            "Wrong answer example:\n" + str(match_list) + f"\nWhy it's wrong:\nMismatch: Expected {expected_rows} rows, but got {df_table.shape[0]} extracted matches."
        )
    
    return dataframe_to_markdown(df_table)
