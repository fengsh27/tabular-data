from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd
import logging

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pe_study_outcome_ver2.pe_study_out_common_agent import (
    PEStudyOutCommonAgentResult,
)

logger = logging.getLogger(__name__)

COLUMN_NUMBER = 3

PARAMETER_VALUE_PROMPT = ChatPromptTemplate.from_template("""
The following main table contains Pharmacoepidemiology data:  
{processed_md_table}
Here is the table caption:  
{caption}
From the main table above, I have extracted a few numerical values to create Subtable 1:  
Below is Subtable 1:
{processed_md_table_with_1_value}
Please review the information in Subtable 1 row by row and complete Subtable 2 accordingly.
Specifically, you need to locate each row from Subtable 1 in the main table, understand its context and meaning, and then use this understanding to populate Subtable 2.
Pay special attention to how the values in the main table relate to both the row and column headers — this will often determine what should be classified as Characteristic, Exposure, or Outcome.

Subtable 2 should include the following column headers only:
**Characteristic, Exposure, Outcome**

    - **Characteristic**: Any geographic, demographic or biological feature of the subjects in the study (e.g., age, sex, race, weight, genetic markers).
    - **Exposure**: Any factor that might be associated with an outcome of interest (e.g., drugs, medical conditions, medications, etc.). If the column header name includes a drug name, it is very likely to be classified as "Exposure".
    - **Outcome**: A measure or set of measures that reflect the effects or consequences of the exposure (e.g., drug treatment, condition, or intervention). These are typically endpoints used to assess the impact of the exposure on the subjects, such as birth weight, symptom reduction, or lab results. In the context of the study outcomes sheet, this category should include all such relevant measures and their associated statistics.    
    
Please Note:
(1) Important: Every row in Subtable 2 must contain exactly {COLUMN_NUMBER} values.
    - Even if you don’t know the value for some columns, you must still fill them with "N/A".
    - Rows with fewer than {COLUMN_NUMBER} values will be considered invalid.
(2) Strictly ensure that you process only rows 0 to {md_table_with_1_value_max_row_index} from the Subtable 1 (which has {md_table_with_1_value_rows} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(3) Carefully review both the row and column headers in the main table, as they are often directly relevant to completing the Characteristic, Exposure, and Outcome columns — or they may provide important context or hints.
(4) **Important:** Please return Subtable 2 as a list of lists, excluding the headers. 
(5) Note: The Outcome column is intended to describe what the main value represents — that is, it conveys the meaning or purpose of the value extracted from the main table. It should not contain the numerical value itself. Only the contextual outcome that the value measures should be included.
(6) The final list should be like this:
[["infants of substance abuse mothers", "cocaine unexposed", "total sleep time"], ["infants of substance abuse mothers", "cocaine exposed", "total sleep time"]]
""")


def get_study_info_prompt(
    md_table: str,
    md_table_with_1_param_type_and_value: str,
    caption: str,
):
    rows_num = markdown_to_dataframe(
        md_table_with_1_param_type_and_value
    ).shape[0]
    return PARAMETER_VALUE_PROMPT.format(
        processed_md_table=display_md_table(md_table),
        caption=caption,
        processed_md_table_with_1_value=display_md_table(
            md_table_with_1_param_type_and_value
        ),
        md_table_with_1_value_max_row_index=rows_num - 1,
        md_table_with_1_value_rows=rows_num,
        COLUMN_NUMBER=COLUMN_NUMBER,
    )


class StudyInfoResult(PEStudyOutCommonAgentResult):
    """Study Info Extraction Result"""

    extracted_study_info: list[list[str]] = Field(
        description="""a list of lists containing parameter values, like 
[["infants of substance abuse mothers", "cocaine unexposed", "total sleep time"], ["infants of substance abuse mothers", "cocaine exposed", "total sleep time"]]"""
    )


def post_process_matched_list(
    res: StudyInfoResult,
    expected_rows: int,
) -> str:
    matched_values = res.extracted_study_info

    # validation
    if not matched_values:
        logger.error("Study info extraction failed: No valid values found.")
        raise ValueError("Study info extraction failed: No valid values found.")

    for item in matched_values:
        if len(item) != COLUMN_NUMBER:
            error_msg = f"""Wrong answer example: 
{str(res.extracted_study_info)}
Why it's wrong. Mismatch: Expected {COLUMN_NUMBER} columns, but got {len(item)} extracted values.
Please make sure the inner list have {COLUMN_NUMBER} values, the result should be like this: 
[["infants of substance abuse mothers", "cocaine unexposed", "total sleep time"], ["infants of substance abuse mothers", "cocaine exposed", "total sleep time"]]
"""
            logger.error(error_msg)
            raise RetryException(error_msg)

    df_table = pd.DataFrame(
        matched_values,
        columns=[
            "Characteristic",
            "Exposure",
            "Outcome",
        ],
    )
    if df_table.shape[0] != expected_rows:
        logger.error(
            "Wrong answer example:\n"
            + str(res.extracted_study_info)
            + f"\nWhy it's wrong:\nMismatch: Expected {expected_rows} rows, but got {df_table.shape[0]} extracted values."
        )
        raise RetryException(
            "Wrong answer example:\n"
            + str(res.extracted_study_info)
            + f"\nWhy it's wrong:\nMismatch: Expected {expected_rows} rows, but got {df_table.shape[0]} extracted values."
        )
    return dataframe_to_markdown(df_table)
