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

(1) Determine how many unique combinations of [Specimen, Sample N, Sample time, Population, Population N] appear in the table.  
    - **Specimen**: The type of biological sample collected (e.g., urine, blood).
    - **Sample N**: The number of samples analyzed for the corresponding specimen.
    - **Sample time:** The specific moment (numerical or time range) when the specimen is sampled.   
    - **Population**: The group of individuals the samples were collected from (e.g., healthy adults, pregnant women).
    - **Population N**: The number of individuals in that population group.
(2) List each unique combination in Python list-of-lists syntax, like this:  
    [["Urine", "20", "... the sentence from the article ...", "Pregnancy", "10"], ["Urine", "20", "... the sentence from the article ...", "Postpregnancy", "10"]] (example)  
(3) Confirm the source of each [Specimen, Sample N, Sample time, Population, Population N] combination before including it in your answer.
(4) In particular, regarding Sample N, please clarify the basis for each value you selected. If there are multiple Sample N values mentioned in different parts of the text, each must be explicitly stated in the original text and should not be derived through calculation or inference. Please cite the exact sentence(s) from the paragraph that support each value.
(5) If both individual Sample N values (e.g., for specific timepoints or population subgroups) and a summed total are reported in the text, only include the individual values. Do not include the summed total, even if it is explicitly stated, to avoid duplication or overcounting.
    For example, if the text states “16 samples were collected in the first trimester, 18 in the second trimester, and a total of 34 across both,” only report the 16 and 18, and exclude the total of 34.
""")


INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."


class SpecimenInfoResult(PKSpecSumCommonAgentResult):
    """Specimen Information Result"""

    specimen_combinations: list[list[str]] = Field(
        description="a list of lists of unique combinations [Specimen, Sample N, Sample time, Population, Population N]"
    )


def post_process_specimen_info(
    res: SpecimenInfoResult,
):
    if res.specimen_combinations is None:
        raise ValueError("Empty specimen combinations")

    if type(res.specimen_combinations) != list or len(res.specimen_combinations) == 0:
        raise RetryException(f"""
Wrong answer: {res.specimen_combinations}, if the table does not explicitly mention the Specimen, Population, please leave it with [["N/A", "N/A", "N/A", "N/A", "N/A"]].
""")

    df_table = pd.DataFrame(
        res.specimen_combinations, columns=["Specimen", "Sample N", "Sample time", "Population", "Population N"]
    )
    return dataframe_to_markdown(df_table)
