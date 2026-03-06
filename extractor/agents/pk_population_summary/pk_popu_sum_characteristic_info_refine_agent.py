from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd
import logging

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_population_summary.pk_popu_sum_common_agent import (
    PKPopuSumCommonAgentResult,
)

logger = logging.getLogger(__name__)

DRUG_INFO_REFINE_PROMPT = ChatPromptTemplate.from_template("""
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Population characteristic" - "Characteristic sub-category" - "Characteristic values" - "Population" - "Population N" - "Source text" as follows:
{processed_md_table_characteristic}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Main value, Unit, Statistics type, Variation type, Variation value, Interval type, Lower bound, Upper bound]** from the table.
    - Main value: The primary value of the characteristic. If there is no variation value or interval, use the value from “Characteristic values” directly.
        For example, if the characteristic value is a ratio like "4/5/4/3", which doesn't follow a standard statistical format, simply enter "4/5/4/3" as the Main value.
    - Unit: The measurement unit of the Main value.
    - Statistics type: the statistics method to summary the Main value, like 'Mean,' 'Median,' 'Count,' etc. **This column is required and must be filled in.**
    - Variation type: the variability measure (describes how spread out the data is) associated with the Main value, like 'Standard Deviation (SD),' 'Proportion (%),' etc.
    - Variation value: the value (not a range) that corresponds to the specific variation.
    - Interval type: the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like 'Minmax,' 'IQR,' etc.
    - Lower bound: the lower bound value of the interval.
    - Upper bound: is the upper bound value of the interval.
    
(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
{{"refined_characteristic_combinations": [["25.4", "year", "Mean", "SD", "0.5", "Minmax", "23.0", "26.1"], ...]}} (example)

(3) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(4) Strictly ensure that you process only rows 0 to {md_table_characteristic_max_row_index} from the Subtable 1 (which has {md_table_characteristic_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Population N for each row in Subtable 2 must be the same as in Subtable 1.
""")


def get_characteristic_info_refine_prompt(title: str, full_text: str, md_table_characteristic: str):
    row_num = markdown_to_dataframe(md_table_characteristic).shape[0]
    return DRUG_INFO_REFINE_PROMPT.format(
        title=title,
        full_text=full_text,
        processed_md_table_characteristic=display_md_table(md_table_characteristic),
        md_table_characteristic_max_row_index=row_num - 1,
        md_table_characteristic_row_num=row_num,
    )


class CharacteristicInfoRefinedResult(PKPopuSumCommonAgentResult):
    """Refined Characteristic Info Result"""

    refined_characteristic_combinations: list[list[str]] = Field(
        description="a list of lists of unique combinations [Main value, Unit, Statistics type, Variation type, Variation value, Interval type, Lower bound, Upper bound]"
    )


def post_process_refined_characteristic_info(
    res: CharacteristicInfoRefinedResult,
    md_table_characteristic: str,
) -> str:
    match_list = res.refined_characteristic_combinations
    if not match_list:
        error_msg = "Characteristic information refinement failed: No valid entries found!"
        logger.error(error_msg)
        raise ValueError(error_msg)

    expected_rows = markdown_to_dataframe(md_table_characteristic).shape[0]
    if len(match_list) != expected_rows:
        error_msg = (
            "Wrong answer example:\n"
            + str(match_list)
            + f"\nWhy it's wrong:\nMismatch: Expected {expected_rows} rows, but got {len(match_list)} extracted matches."
        )
        logger.error(error_msg)
        raise RetryException(error_msg)

    df_table = pd.DataFrame(
        match_list,
        columns=["Main value", "Unit", "Statistics type", "Variation type", "Variation value", "Interval type", "Lower bound", "Upper bound"],
    ).astype(str)

    # df_drug = markdown_to_dataframe(md_table_characteristic)
    # if not df_table["Population N"].equals(df_drug["Population N"]):
    #     error_msg = (
    #         "Wrong answer example:\n"
    #         + str(match_list)
    #         + "\nWhy it's wrong:\nThe rows in the refined Subtable 2 do not correspond to those in Subtable 1 on a one-to-one basis."
    #     )
    #     if df_drug.shape[0] == df_table.shape[0]:
    #         # check row by row
    #         list1 = df_table["Population N"].to_list()
    #         list_patient = df_drug["Population N"].to_list()
    #         # check row by row
    #         for ix in range(len(list1)):
    #             item1: str = list1[ix]
    #             item2: str = list_patient[ix]
    #             item1 = item1.strip().strip("\"'")
    #             item2 = item2.strip().strip("\"'")
    #             if item1 == item2:
    #                 continue
    #             logger.error(
    #                 error_msg + f"\nExpedted df_drug['Population N']: {list_patient}"
    #             )
    #             raise RetryException(error_msg)
    #     else:
    #         logger.error(
    #             error_msg + f"\nExpedted df_drug['Population N']: {list_patient}"
    #         )
    #         raise RetryException(error_msg)

    return dataframe_to_markdown(df_table)
