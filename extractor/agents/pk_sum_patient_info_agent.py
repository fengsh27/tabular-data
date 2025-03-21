
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_sum_common_agent import PKSumCommonAgentResult

PATIENT_INFO_PROMPT = ChatPromptTemplate.from_template("""
the following table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
Carefully analyze the table, **row by row and column by column**, and follow these steps:
(1) Identify how many unique [Population, Pregnancy stage, Subject N] combinations are present in the table.
Population is the patient age group.
Pregnancy stage is the pregnancy stages of patients mentioned in the study.
Subject N represents the number of subjects corresponding to the specific parameter or the number of samples with quantifiable levels of the respective analyte.
(2) List each unique combination in the format of a list of lists in one line, using Python string syntax. Your answer should be enclosed in double angle brackets <<>>.
(3) Ensure that all elements in the list of lists are **strings**, especially Subject N, which must be enclosed in double quotes (`""`).
(4) Verify the source of each [Population, Pregnancy stage, Subject N] combination before including it in your answer.
(5) The "Subject N" values within each population group sometimes differ slightly across parameters. This reflects data availability for each specific parameter within that age group. **YOU MUST** include all the Ns for each age group.
    - Specifically, **YOU MUST** explain every number, in this list: {int_list} to determine if it should be listed in Subject N.
    - For example, if a population group has a Subject N of 8, but further analysis shows that 5, 6, and 7 of the 8 subjects correspond to different parameter values, then 5, 6, 7, and 8 must all be included as Subject N in different combinations in the final answer.
    - Fill in "N/A" when you don't know the exact N.
(6) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred.
""")

INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

class PatientInfoResult(PKSumCommonAgentResult):
    """ Patient Information Result """
    patient_combinations: List[List[str]] = Field(description="a list of lists of unique combinations [Population, Pregnancy stage, Subject N]")
    

def post_process_convert_patient_info_to_md_table(
    res: PatientInfoResult,
) -> str:
    match_list = res.patient_combinations
    match_list = [list(t) for t in dict.fromkeys(map(tuple, match_list))]

    df_table = pd.DataFrame(match_list, columns=["Population", "Pregnancy stage", "Subject N"])
    return_md_table = dataframe_to_markdown(df_table)
    return return_md_table

