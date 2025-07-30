from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd
import logging

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pe_study_outcome.pe_study_out_common_agent import (
    PEStudyOutCommonAgentResult,
)

logger = logging.getLogger(__name__)

CHARACTERISTIC_REFINE_PROMPT = ChatPromptTemplate.from_template("""
We are processing an article: "{article_title}"
The following main table contains Pharmacoepidemiology data:  
{processed_md_table}
Here is the table caption:  
{caption}
From the main table above, I have extracted a few numerical values to create Subtable 1:  
Below is Subtable 1:
{processed_md_table_with_1_value}
Please review the information in Subtable 1 row by row and complete Subtable 2 accordingly.
Specifically, you need to interpret the meaning of each entry in the "Value" column of Subtable 1 and rewrite it in a more structured and standardized format in Subtable 2.
Subtable 2 should include the following column headers only:
**Main value, Main value unit, Statistics type, Variation type, Variation value, Interval type, Lower bound, Upper bound**

Main value: the value of main parameter (not a range). 
Main value unit: The unit of the main parameter (e.g. kg, g, Count) **DO NOT USE Statistics type, such as SD, as the unit!!**
Statistics type: The statistical method used to summarize the Main value, such as Mean, Median, Sum, Proportion, or %, etc. This column is required and must be completed.
Variation type: the variability measure (describes how spread out the data is) associated with the Main value, like 'Standard Deviation (SD),' etc.
Variation value: the value (not a range) that corresponds to the specific variation.
    **Please note:** In addition to common cases like standard deviations (SD), there is a special case that should also be handled using the Variation type and Variation value columns:
    Often, datasets report both count and percentage values together. In such cases, enter the count into Main value, set Main value unit to "Count", and choose "Sum" for Statistics type. 
    Then, record the percentage or proportion under Variation type and Variation value, respectively.

Interval type: the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like '95% CI,' 'Range,' 'IQR,' etc.
Lower bound: the lower bound value of the interval.
Upper bound: is the upper bound value of the interval.

Please Note:
(1) An interval consisting of two numbers must be placed separately into the Low limit and High limit fields; it is prohibited to place it in the Variation value field.
(2) For values that do not need to be filled, enter "N/A".
(3) Strictly ensure that you process only rows 0 to {md_table_with_1_value_max_row_index} from the Subtable 1 (which has {md_table_with_1_value_rows} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) For rows in Subtable 1 that can not be extracted, enter "N/A" for the entire row.
(3) **Important:** Please return Subtable 2 as a list of lists, excluding the headers. Ensure all values are converted to strings.
(4) **Absolutely no calculations are allowed—every value must be taken directly from Subtable 1 without any modifications.**  
(5) The final list should be like this:
[["10", "Count", "Sum", "%", "1", "N/A", "N/A", "N/A"], ["50", "N/A", "%", "N/A", "N/A", "N/A", "N/A", "N/A"]]
""")


def get_characteristic_refine_prompt(
    md_table: str,
    md_table_with_1_param_type_and_value: str,
    caption: str,
):
    rows_num = markdown_to_dataframe(
        md_table_with_1_param_type_and_value
    ).shape[0]
    return CHARACTERISTIC_REFINE_PROMPT.format(
        processed_md_table=display_md_table(md_table),
        caption=caption,
        processed_md_table_with_1_value=display_md_table(
            md_table_with_1_param_type_and_value
        ),
        md_table_with_1_value_max_row_index=rows_num - 1,
        md_table_with_1_value_rows=rows_num,
    )


COLUMN_NUMBER = 8


class CharacteristicRefinementResult(PEStudyOutCommonAgentResult):

    extracted_param_values: list[list[str]] = Field(
        description="""a list of lists containing parameter values, like 
[["10", "Count", "Sum", "%", "1", "N/A", "N/A", "N/A"], ["50", "N/A", "%", "N/A", "N/A", "N/A", "N/A", "N/A"]]"""
    )


def post_process_matched_list(
    res: CharacteristicRefinementResult,
    expected_rows: int,
) -> str:
    matched_values = res.extracted_param_values

    # validation
    if not matched_values:
        logger.error("Parameter value extraction failed: No valid values found.")
        raise ValueError("Parameter value extraction failed: No valid values found.")

    for item in matched_values:
        if len(item) != COLUMN_NUMBER:
            error_msg = f"""Wrong answer example: 
{str(res.extracted_param_values)}
Why it's wrong. Mismatch: Expected {COLUMN_NUMBER} columns, but got {len(item)} extracted values.
Please make sure the inner list have {COLUMN_NUMBER} values, the result should be like this: 
[["10", "Count", "Sum", "%", "1", "N/A", "N/A", "N/A"], ["50", "N/A", "%", "N/A", "N/A", "N/A", "N/A", "N/A"]]
"""
            logger.error(error_msg)
            raise RetryException(error_msg)

    df_table = pd.DataFrame(
        matched_values,
        columns=[
            "Main value",
            "Main value unit",
            "Statistics type",
            "Variation type",
            "Variation value",
            "Interval type",
            "Lower bound",
            "Upper bound"
        ],
    )
    if df_table.shape[0] != expected_rows:
        logger.error(
            "Wrong answer example:\n"
            + str(res.extracted_param_values)
            + f"\nWhy it's wrong:\nMismatch: Expected {expected_rows} rows, but got {df_table.shape[0]} extracted values."
        )
        raise RetryException(
            "Wrong answer example:\n"
            + str(res.extracted_param_values)
            + f"\nWhy it's wrong:\nMismatch: Expected {expected_rows} rows, but got {df_table.shape[0]} extracted values."
        )
    return dataframe_to_markdown(df_table)
