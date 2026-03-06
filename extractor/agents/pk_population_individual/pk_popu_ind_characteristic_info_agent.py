from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_population_individual.pk_popu_ind_common_agent import (
    PKPopuIndCommonAgentResult,
)

CHARACTERISTIC_INFO_PROMPT = ChatPromptTemplate.from_template("""
{title}
{full_text}
Read the article and answer the following:

(1) Determine how many unique combinations of [Patient ID, Patient characteristic, Characteristic sub-category, Characteristic values, Source text] appear in the table.  
    - **Patient ID**: Patient ID refers to the identifier assigned to each patient.
    - **Patient characteristic**: Patient-focused characteristics. Not PK parameter!!!
            · “Age," “Sex," "Weight," “Gender," “Race," “Ethnicity"
            · “Socioeconomic status," “Education," “Marital status"
            · “Comorbidity," “Drug indication," “Adverse events"
            · “Severity," “BMI," “Smoking status," “Alcohol use," "Blood pressure"
    - **Characteristic sub-category**: Levels or options under characteristics.
            · For sex: “Male", “Female"
            · For race: “White", “Black", “Asian", “Hispanic"
            · For comorbidity: “Diabetes", “Hypertension", “Asthma"
            · For adverse events: “Mild", “Moderate", “Severe"
            · If no sub-category, use "N/A"
    - **Characteristic values**: The numerical descriptor. 
    - **Source text**: The original sentence or excerpt from the source document where the data was reported. This field provides context and traceability, ensuring that each data point can be verified against its original description in the literature. Use "N/A" if no source can be found.
(2) List each unique combination in Python list-of-lists syntax, like this:  
    {{"characteristic_combinations": [["1", "Weight", "N/A", "76.8", "... the sentence from the article ..."], ["2", "Age", "N/A", "23", "... the sentence from the article ..."]]}} (example)  
(3) Confirm the source of each [Patient ID, Patient characteristic, Characteristic sub-category, Characteristic values, Source text] combination before including it in your answer.
""")


INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."


class CharacteristicInfoResult(PKPopuIndCommonAgentResult):
    """Specimen Information Result"""

    characteristic_combinations: list[list[str]] = Field(
        description="a list of lists of unique combinations [Patient ID, Patient characteristic, Characteristic sub-category, Characteristic values, Source text]"
    )


def post_process_characteristic_info(
    res: CharacteristicInfoResult,
):
    if res.characteristic_combinations is None:
        raise ValueError("Empty characteristic combinations")

    if type(res.characteristic_combinations) != list or len(res.characteristic_combinations) == 0:
        raise RetryException(f"""
Wrong answer: {res.characteristic_combinations}, if the table does not explicitly mention any [Population characteristic, Characteristic sub-category, Characteristic values, Population, Population N, Source text], please leave it with [["N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]].
""")

    df_table = pd.DataFrame(
        res.characteristic_combinations, columns=["Patient ID", "Patient characteristic", "Characteristic sub-category", "Characteristic values", "Source text"]
    )

    if "|" in dataframe_to_markdown(df_table):
        for row_idx in df_table.index:
            for col in df_table.columns:
                cell = df_table.at[row_idx, col]
                if isinstance(cell, str) and "|" in cell:
                    updated = "Content from Table: " + cell.replace("|", "-")
                    df_table.at[row_idx, col] = updated

    return dataframe_to_markdown(df_table)
