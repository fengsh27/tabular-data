from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgentResult

PATIENT_INFO_PROMPT = ChatPromptTemplate.from_template("""
You are a **pharmacokinetics (PK) domain expert** specializing in structured data extraction from biomedical tables.

You are provided with:

* A pharmacokinetics (PK) data table in Markdown format:
{processed_md_table}
* The table caption:
{caption}

Your task is to **systematically extract population-level identifiers** from the table by carefully examining it **row by row and column by column**.

---

### **Extraction Task**

#### **Step 1: Identify Unique Combinations**

Determine all **unique combinations** of the following three elements present in the table:

1. **Patient ID**

   * Refers to an identifier explicitly assigned to a **unique individual patient** in the table.
   * Use the **exact text** as it appears in the table.
   * If no explicit individual patient identifier exists:

     * Infer a **unique unit** that clearly distinguishes rows (e.g., subject number, case number, mother–infant pair, cohort label).
     * Use this inferred unit as the Patient ID.
   * If **neither an individual patient nor a reasonable unique unit can be identified**, do **not** fabricate identifiers.

2. **Population**

   * Refers to the **age-based or demographic population group** (e.g., adult, neonate, pregnant women).
   * If not explicitly stated or reasonably inferable, use `"N/A"`.

3. **Pregnancy Stage**

   * Refers to pregnancy-related timing or stage (e.g., trimester, delivery, postpartum).
   * If not explicitly stated or reasonably inferable, use `"N/A"`.

---

#### **Step 2: Validation Rules**

* Verify that **each extracted combination is supported by the table content or caption**.
* If information is missing:

  * First attempt to infer it using contextual clues within the table, caption, or standard PK study conventions.
  * Use `"N/A"` **only if inference is not reasonably possible**.
* If you **cannot identify any individual Patient ID or infer any unique unit**, return an **empty list** and do **not** guess or fabricate data.

---

### **Output Requirements**

Your response **must be valid JSON** and **must exactly match** the structure below:

```json
{{
  "reasoning_process": "<1–2 concise sentences summarizing how the combinations were identified>",
  "patient_combinations": [
    ["Patient ID", "Population", "Pregnancy stage"]
  ]
}}
```

#### **Strict Formatting Rules**

* `patient_combinations` must be a **list of lists**.
* **All elements must be strings**, including `"N/A"`.
* **Patient ID values must be enclosed in double quotes**.
* Do **not** include any explanatory text outside the JSON object.
* Do **not** include Markdown, comments, or code fences.

---

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
