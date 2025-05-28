from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pk_specimen_summary.pk_spec_sum_common_agent import (
    PKSpecSumCommonAgentResult,
    RetryException,
)

SPECIMEN_INFO_PROMPT = ChatPromptTemplate.from_template("""
{title}
{full_text}
Read the article and answer the following:

(1) Determine how many unique combinations of [Patient ID, Specimen, Sample N, Sample time] appear in the table.  
    - **Patient ID**: Patient ID refers to the identifier assigned to each individual patient.
    - **Specimen**: The type of biological sample collected (e.g., urine, blood).
    - **Sample N**: The number of samples analyzed for the corresponding specimen.
    - **Sample time:** The specific moment (numerical or time range) when the specimen is sampled.   
(2) List each unique combination in Python list-of-lists syntax, like this:  
    [["1", "Urine", "20", "... the sentence from the article ..."], ["2", "Urine", "20", "... the sentence from the article ..."]] (example)  
(3) Confirm the source of each [Patient ID, Specimen, Sample N, Sample time] combination before including it in your answer.
""")


INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."


class SpecimenInfoResult(PKSpecSumCommonAgentResult):
    """Specimen Information Result"""

    specimen_combinations: list[list[str]] = Field(
        description="a list of lists of unique combinations [Patient ID, Specimen, Sample N, Sample time]"
    )


def post_process_specimen_info(
    res: SpecimenInfoResult,
):
    if res.specimen_combinations is None:
        raise ValueError("Empty specimen combinations")

    if type(res.specimen_combinations) != list or len(res.specimen_combinations) == 0:
        raise RetryException(f"""
Wrong answer: {res.specimen_combinations}, if the table does not explicitly mention any [Specimen, Sample N, Sample time, Population, Population N], please leave it with [["N/A", "N/A", "N/A", "N/A", "N/A"]].
""")

    df_table = pd.DataFrame(
        res.specimen_combinations, columns=["Patient ID", "Specimen", "Sample N", "Sample time"]
    )
    return dataframe_to_markdown(df_table)
