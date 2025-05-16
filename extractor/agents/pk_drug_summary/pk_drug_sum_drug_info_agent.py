from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pk_drug_summary.pk_drug_sum_common_agent import (
    PKDrugSumCommonAgentResult,
    RetryException,
)

DRUG_INFO_PROMPT = ChatPromptTemplate.from_template("""
{title}
{full_text}
Read the article and answer the following:

(1) Determine how many unique combinations of [Drug/Metabolite name, Dose frequency, Dose amount, Population, Population N] appear in the table.  
    - **Drug/Metabolite name**: The name of the drug or its metabolite that has been studied.
    - **Dose frequency**: The number of times the drug was taken. (e.g., Single, Multiple, 3, 4)
    - **Dose amount**: The amount of drug, a value, a list, or a range, (e.g., 5 mg; 1,2,3,4 g; 0.01 - 0.05 mg) each time the drug was taken.  
    - **Population**: The group of individuals the samples were collected from (e.g., healthy adults, pregnant women).
    - **Population N**: The number of individuals in that population group.
    - **Source text**: The original sentence or excerpt from the source document where the data was reported. This field provides context and traceability, ensuring that each data point can be verified against its original description in the literature. Use "N/A" if no source can be found.
(2) List each unique combination in Python list-of-lists syntax, like this:  
    [["Lorazepam", "2 doses", "0.01 mg", "Pregnancy", "10", "...the source text..."], ["Fentanyl", "Single dose", "0.01 mg", "Postpregnancy", "10", "...the source text..."]] (example)  
(3) Confirm the source of each [Drug/Metabolite name, Dose frequency, Dose amount, Population, Drug N] combination before including it in your answer.
""")


INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."


class DrugInfoResult(PKDrugSumCommonAgentResult):
    """Drug Information Result"""

    population_combinations: list[list[str]] = Field(
        description="a list of lists of unique combinations [Drug/Metabolite name, Dose frequency, Dose amount, Population, Population N, Source text]"
    )


def post_process_population_info(
    res: DrugInfoResult,
):
    if res.population_combinations is None:
        raise ValueError("Empty population combinations")

    if type(res.population_combinations) != list or len(res.population_combinations) == 0:
        raise RetryException(f"""
Wrong answer: {res.population_combinations}, if the table does not explicitly mention any [Drug/Metabolite name, Dose frequency, Dose amount, Population, Population N, Source text], please leave it with [["N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]].
""")

    df_table = pd.DataFrame(
        res.population_combinations, columns=["Drug/Metabolite name", "Dose frequency", "Dose amount", "Population", "Population N", "Source text"]
    )
    return dataframe_to_markdown(df_table)
