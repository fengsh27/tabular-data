from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgentResult

PATIENT_INFO_PROMPT = ChatPromptTemplate.from_template("""
the following table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
Carefully analyze the table, **row by row and column by column**, and follow these steps:
(1) Identify how many unique [Patient ID, Population, Pregnancy stage] combinations are present in the table.
Patient ID refers to the identifier assigned to each patient.
Population is the patient age group.
Pregnancy stage is the pregnancy stages of patients mentioned in the study.
(2) List each unique combination in the format of a list of lists in one line, using Python string syntax. Your answer should be enclosed in double angle brackets <<>>.
(3) Ensure that all elements in the list of lists are **strings**, especially Patient ID, which must be enclosed in double quotes (`""`).
(4) Verify the source of each [Patient ID, Population, Pregnancy stage] combination before including it in your answer.
(5) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred.
""")

INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."


class PatientInfoResult(PKIndCommonAgentResult):
    """Patient Information Result"""

    patient_combinations: list[list[str]] = Field(
        description="a list of lists of unique combinations [Patient ID, Population, Pregnancy stage]"
    )


def post_process_convert_patient_info_to_md_table(
    res: PatientInfoResult,
) -> str:
    match_list = res.patient_combinations
    match_list = [list(t) for t in dict.fromkeys(map(tuple, match_list))]

    df_table = pd.DataFrame(
        match_list, columns=["Patient ID", "Population", "Pregnancy stage"]
    )
    return_md_table = dataframe_to_markdown(df_table)
    return return_md_table
