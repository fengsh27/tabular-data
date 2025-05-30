from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pk_individual.pk_ind_common_agent import (
    PKIndCommonAgentResult,
    RetryException,
)

DRUG_INFO_PROMPT = ChatPromptTemplate.from_template("""
The following table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
This table is from a paper titled: **{paper_title}**
Carefully analyze the paper title and the table and follow these steps:  
(1) Identify how many unique [Drug name, Analyte, Specimen] combinations are present in the table.  
Drug name is the name of the drug mentioned in the study.
Analyte is the substance measured in the study, which can be the primary drug, its metabolite, or another drug it affects, etc. When filling in "Analyte," only enter the name of the substance.
Specimen is the type of sample.
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be enclosed in double angle brackets, like this:  
   [["Lorazepam", "Lorazepam", "Plasma"], ["Lorazepam", "Lorazepam", "Urine"]] (example)  
(3) Verify the source of each [Drug Name, Analyte, Specimen] combination before including it in your answer.  
(4) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred.
(5) If none of the elements are explicitly stated in the table or caption, infer them from the **paper title**.
""")

INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."


class DrugInfoResult(PKIndCommonAgentResult):
    """Drug Information Result"""

    drug_combinations: list[list[str]] = Field(
        description="a list of lists of unique combinations [Drug name, Analyte, Specimen]"
    )


def post_process_drug_info(
    res: DrugInfoResult,
):
    if res.drug_combinations is None:
        raise ValueError("Empty drug combinations")

    if type(res.drug_combinations) != list or len(res.drug_combinations) == 0:
        raise RetryException(f"""
Wrong answer: {res.drug_combinations}, if the table does not explicitly mention the drug name, analyte, please leave it with [["N/A", "N/A", "N/A"]].
""")

    df_table = pd.DataFrame(
        res.drug_combinations, columns=["Drug name", "Analyte", "Specimen"]
    )
    return dataframe_to_markdown(df_table)
