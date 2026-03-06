from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd
import logging

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pe_study_info.pe_study_info_common_agent import (
    PEStudyInfoCommonAgentResult,
)

logger = logging.getLogger(__name__)

DESIGN_INFO_REFINE_PROMPT = ChatPromptTemplate.from_template("""
{title}
{full_text}
Read the article and answer the following:

Based on the article, Subtable 1 has already been created by extracting and combining information on "Study type" – "Study design" – "Data source" into a one-row format.
{processed_md_table_design}
Now, please carefully review the article and extract the Study-designn-related information to create Subtable 2, also in a one-row format. Subtable 2 should complement and enhance the information in Subtable 1.

(1) Identify all unique combinations of **[Population, Inclusion criteria, Exclusion criteria, Pregnancy stage, Subject N, Drug name, Outcomes]** from the table.
    - **Population**: The age group of the subjects.  
        **Common categories include:**  
            - "Nonpregnant"
            - "Maternal" (pregnant individuals)
            - "Pediatric" (generally birth to ~17 years)  
            - "Adults" (typically 18 years or older)  
    - **Inclusion criteria**: Characteristics that study participants must have to be qualified to be part of a study. (use exact wording from the article)
    - **Exclusion criteria**: Characteristics that disqualify interested participants from participating in a study. (use exact wording from the article)
    - **Pregnancy stage**: The stage of pregnancy for the patients in the study.  
        **Common categories include:**  
            - "Trimester 1" (usually up to 14 weeks of pregnancy)  
            - "Trimester 2" (~15–28 weeks of pregnancy)  
            - "Trimester 3" (~≥ 28 weeks of pregnancy)  
            - "Fetus" or "Fetal Stage" (referring to the developing baby during pregnancy)  
            - "Parturition," "Labor," or "Delivery" (the process of childbirth)  
            - "Postpartum" (~6–8 weeks after birth)  
            - "Nursing," "Breastfeeding," or "Lactation" (refers to the period of breastfeeding after birth) 
    - **Subject N**: The number of subjects corresponding to the specific population.
    - **Drug name**: All drugs of interest as they relate to the outcomes of a particular study.
    - **Outcomes**: A measure(s) of interest that an investigator(s) considers the most important among the many outcomes to be examined in the study. (use exact wording from the article)

(2) Write the one-row subtable 2 into the format of a **list of lists**, using **Python string syntax**. The result should be like this:
{{"refined_design_combinations": [["Maternal", "Inclusion criteria should be retrieved from the article, use the exact wording from the article", "Exclusion criteria should be retrieved from the article, use the exact wording from the article", "Trimester 3", "20", "Lorazepam", "Outcomes should be retrieved from the article, use the exact wording from the article"]]}} (example)

(3) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
""")


def get_design_info_refine_prompt(title: str, full_text: str, md_table_design: str):
    row_num = markdown_to_dataframe(md_table_design).shape[0]
    return DESIGN_INFO_REFINE_PROMPT.format(
        title=title,
        full_text=full_text,
        processed_md_table_design=display_md_table(md_table_design),
        md_table_design_max_row_index=row_num - 1,
        md_table_design_row_num=row_num,
    )


class DesignInfoRefinedResult(PEStudyInfoCommonAgentResult):
    """Refined Patient Info Result"""

    refined_design_combinations: list[list[str]] = Field(
        description="a list of lists, but only has one combination of [Population, Inclusion criteria, Exclusion criteria, Pregnancy stage, Subject N, Drug name, Outcomes]"
    )


def post_process_refined_design_info(
    res: DesignInfoRefinedResult,
    md_table_design: str,
) -> str:
    match_list = res.refined_design_combinations
    if not match_list:
        error_msg = "Design information refinement failed: No valid entries found!"
        logger.error(error_msg)
        raise ValueError(error_msg)

    expected_rows = markdown_to_dataframe(md_table_design).shape[0]
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
            "Inclusion criteria",
            "Exclusion criteria",
            "Pregnancy stage",
            "Subject N",
            "Drug name",
            "Outcomes",
        ],
    ).astype(str)

    if "|" in dataframe_to_markdown(df_table):
        for row_idx in df_table.index:
            for col in df_table.columns:
                cell = df_table.at[row_idx, col]
                if isinstance(cell, str) and "|" in cell:
                    updated = "Content from Table: " + cell.replace("|", "-")
                    df_table.at[row_idx, col] = updated

    return dataframe_to_markdown(df_table)
