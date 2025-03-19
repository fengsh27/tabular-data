
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

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
Subject N is the number of subjects that correspond to the specific parameter.
(2) List each unique combination in the format of a list of lists in one line, using Python string syntax. Your answer should be enclosed in double angle brackets <<>>. 
(3) Verify the source of each [Population, Pregnancy stage, Subject N] combination before including it in your answer.  
(4) The "Subject N" values within each population group sometimes differ slightly across parameters. This reflects data availability for each specific parameter within that age group. You must include all the Ns for each age group. 
Specifically, make sure to check every number in this list: {int_list} to determine if it should be listed in Subject N. Fill in "N/A" when you don't know the exact N.
(5) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred. 
""")

INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

class PatientInfoResult(PKSumCommonAgentResult):
    """ Patient Information Result """
    patient_combinations: List[List[str]] = Field(description="a list of lists of unique combinations [Population, Pregnancy stage, Subject N]")
    


    

