from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pk_population_summary.pk_popu_sum_common_agent import (
    PKPopuSumCommonAgentResult,
    RetryException,
)

CHARACTERISTIC_INFO_PROMPT = ChatPromptTemplate.from_template("""
{title}
{full_text}
Read the article and answer the following:

(1) Determine how many unique combinations of [Population characteristic, Characteristic sub-category, Characteristic values, Population, Population N, Source text] appear in the table.  
    - **Population characteristic**: Population-focused characteristics. Not PK parameter!!!
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
    - **Characteristic values**: Include all numerical descriptors—such as means, ranges, and p-values. If multiple numerical descriptors are reported, you must include them all.
    - **Population**: The group of individuals the samples were collected from (e.g., healthy adults, pregnant women).
    - **Population N**: The number of individuals in that population group.
    - **Source text**: The original sentence or excerpt from the source document where the data was reported. This field provides context and traceability, ensuring that each data point can be verified against its original description in the literature. Use "N/A" if no source can be found.
(2) List each unique combination in Python list-of-lists syntax, like this:  
    [["Weight", "N/A", "76.8 (67.4-86.2)", "Pregnancy", "10", "... the sentence from the article ..."], ["Age", "N/A", "23.3 (19.02-27.58)", "Postpregnancy", "10", "... the sentence from the article ..."]] (example)  
(3) Confirm the source of each [Population characteristic, Characteristic values, Population, Population N, Source text] combination before including it in your answer.
(4) In particular, regarding Sample N, please clarify the basis for each value you selected. If there are multiple Sample N values mentioned in different parts of the text, each must be explicitly stated in the original text and should not be derived through calculation or inference. Please cite the exact sentence(s) from the paragraph that support each value.
(5) If both individual Sample N values (e.g., for specific timepoints or population subgroups) and a summed total are reported in the text, only include the individual values. Do not include the summed total, even if it is explicitly stated, to avoid duplication or overcounting.
    For example, if the text states “16 samples were collected in the first trimester, 18 in the second trimester, and a total of 34 across both," only report the 16 and 18, and exclude the total of 34.
""")


INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."


class CharacteristicInfoResult(PKPopuSumCommonAgentResult):
    """Specimen Information Result"""

    characteristic_combinations: list[list[str]] = Field(
        description="a list of lists of unique combinations [Population characteristic, Characteristic sub-category, Characteristic values, Population, Population N, Source text]"
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
        res.characteristic_combinations, columns=["Population characteristic", "Characteristic sub-category", "Characteristic values", "Population", "Population N", "Source text"]
    )
    return dataframe_to_markdown(df_table)
