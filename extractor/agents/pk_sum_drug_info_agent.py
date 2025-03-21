
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_sum_common_agent import (
    PKSumCommonAgentResult,
)

DRUG_INFO_PROMPT = ChatPromptTemplate.from_template("""
the following table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
Carefully analyze the table and follow these steps:  
(1) Identify how many unique [Drug name, Analyte, Specimen] combinations are present in the table.  
Drug name is the name of the drug mentioned in the study.
Analyte is the substance measured in the study, which can be the primary drug, its metabolite, or another drug it affects, etc. When filling in "Analyte," only enter the name of the substance.
Specimen is the type of sample.
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be enclosed in double angle brackets, like this:  
   <<[["Lorazepam", "Lorazepam", "Plasma"], ["Lorazepam", "Lorazepam", "Urine"]]>> (example)  
(3) Verify the source of each [Drug Name, Analyte, Specimen] combination before including it in your answer.  
(4) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred.
""")

INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

class DrugInfoResult(PKSumCommonAgentResult):
    """ Drug Information Result """
    drug_combinations: List[List[str]] = Field(description="a list of lists of unique combinations [Drug name, Analyte, Specimen]")
 

def post_process_drug_info(
    res: DrugInfoResult,
):
    if res.drug_combinations is None:
        raise ValueError("Empty drug combinations")
    
    df_table = pd.DataFrame(res.drug_combinations, columns=["Drug name", "Analyte", "Specimen"])
    return dataframe_to_markdown(df_table)

