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

PATIENT_INFO_REFINE_PROMPT = ChatPromptTemplate.from_template("""
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Population characteristic" - "Characteristic sub-category" - "Characteristic values" - "Population" - "Population N" - "Source text" as follows:
{processed_md_table_characteristic}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Population, Pregnancy stage, Pediatric/Gestational age, Population N]** from the table.
    - **Population**: The age group of the subjects.  
      **Common categories include:**  
        - "Nonpregnant"
        - "Maternal" (pregnant individuals)
        - "Pediatric" (generally birth to ~17 years)  
        - "Adults" (typically 18 years or older)  
      
    - **Pregnancy stage**: The stage of pregnancy for the patients in the study.  
      **Common categories include:**  
        - "Pre-pregnancy"
        - "Trimester 1" (usually up to 14 weeks of pregnancy)  
        - "Trimester 2" (~15–28 weeks of pregnancy)  
        - "Trimester 3" (~≥ 28 weeks of pregnancy)  
        - "Fetus" (referring to the developing baby during pregnancy)  
        - "Parturition," "Labor," or "Delivery" (the process of childbirth)  
        - "Postpartum" (~6–8 weeks after birth)  
        - "Nursing," "Breastfeeding," or "Lactation" (refers to the period of breastfeeding after birth) 
 
    - **Pediatric/Gestational age**: The child's age (or age range) at a specific point in the study. Retain the original wording whenever possible. It can also be the pregnancy weeks.
        Note: Verify that the value explicitly states the age. Only consider it valid if the age is directly mentioned. Do not infer age from the timing of data recording or drug administration.
        For example: "Concentrations on Days 7" refers to a measurement time point, not an age, and should not be treated as such.
        
    - **Population N**: The number of people corresponding to the specific population.
    
(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
{{"refined_patient_combinations": [["Maternal", "Trimester 1", "N/A", "15"], ...]}} (example)

(3) For each Population, determine whether it can be classified under one or more of the common categories listed above. If it matches one or more standard categories, replace it with the corresponding standard category (or categories). If it does not fit any common category, retain the original wording.

(4) For each Pregnancy Stage, check whether it aligns with any of the common categories. If it does, replace it with the corresponding standard category. If it does not fit any common category, keep the original wording unchanged.

(5) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(6) Strictly ensure that you process only rows 0 to {md_table_characteristic_max_row_index} from the Subtable 1 (which has {md_table_characteristic_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Population N for each row in Subtable 2 must be the same as in Subtable 1.
""")


def get_patient_info_refine_prompt(title: str, full_text: str, md_table_characteristic: str):
    row_num = markdown_to_dataframe(md_table_characteristic).shape[0]
    return PATIENT_INFO_REFINE_PROMPT.format(
        title=title,
        full_text=full_text,
        processed_md_table_characteristic=display_md_table(md_table_characteristic),
        md_table_characteristic_max_row_index=row_num - 1,
        md_table_characteristic_row_num=row_num,
    )


class PatientInfoRefinedResult(PKPopuSumCommonAgentResult):
    """Refined Patient Info Result"""

    refined_patient_combinations: list[list[str]] = Field(
        description="a list of lists of unique combinations [Population, Pregnancy stage, Pediatric/Gestational age, Population N]"
    )


def post_process_refined_patient_info(
    res: PatientInfoRefinedResult,
    md_table_characteristic: str,
) -> str:
    match_list = res.refined_patient_combinations
    if not match_list:
        error_msg = "Population information refinement failed: No valid entries found!"
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
        columns=[
            "Population",
            "Pregnancy stage",
            "Pediatric/Gestational age",
            "Population N",
        ],
    ).astype(str)

    df_characteristic = markdown_to_dataframe(md_table_characteristic)
    if not df_table["Population N"].equals(df_characteristic["Population N"]):
        error_msg = (
            "Wrong answer example:\n"
            + str(match_list)
            + "\nWhy it's wrong:\nThe rows in the refined Subtable 2 do not correspond to those in Subtable 1 on a one-to-one basis."
        )
        if df_characteristic.shape[0] == df_table.shape[0]:
            # check row by row
            list1 = df_table["Population N"].to_list()
            list_patient = df_characteristic["Population N"].to_list()
            # check row by row
            for ix in range(len(list1)):
                item1: str = list1[ix]
                item2: str = list_patient[ix]
                item1 = item1.strip().strip("\"'")
                item2 = item2.strip().strip("\"'")
                if item1 == item2:
                    continue
                logger.error(
                    error_msg + f"\nExpedted df_characteristic['Population N']: {list_patient}"
                )
                raise RetryException(error_msg)
        else:
            logger.error(
                error_msg + f"\nExpedted df_characteristic['Population N']: {list_patient}"
            )
            raise RetryException(error_msg)

    return dataframe_to_markdown(df_table)
