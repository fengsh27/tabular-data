from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, ValidationError
import logging

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgentResult

logger = logging.getLogger(__name__)

HEADER_CATEGORIZE_PROMPT = ChatPromptTemplate.from_template("""
### **Task**

The following table contains pharmacokinetics (PK) data:

{processed_md_table_aligned}
{column_headers_str}

Each column header is **already defined** and may appear as:

* a single string (e.g., `"Volunteer"`), or
* a multi-level / tuple-style header (e.g., `("Citalopram", "Maximum1 milk concentration (µg l−1)")`).

### **Your task is ONLY to categorize columns — NOT to modify them.**

---

### **Step 1 — Column Categorization**

Examine **each column header exactly as provided** and assign it to **one and only one** of the following categories:

* **"Patient ID"**
  Columns that uniquely identify a patient or participant.
  **At least one column must be assigned as "Patient ID".**

* **"Parameter value"** (Dependent Variable)
  Columns that report pharmacokinetic measurements or ratios, such as concentrations or PK parameters.
  Examples include (but are not limited to):

  * Concentration (e.g., plasma/milk concentration)
  * AUC
  * Cmax
  * Tmax
  * Half-life (T½)
  * Clearance
  * Ratios (e.g., M/P AUC)

* **"Uncategorized"**
  Columns that do not represent PK outcome values or patient identifiers.
  Examples include dose, time, study period, subject number, sampling time, or metadata.

---

### **Step 2 — Special Rules**

1. If a column refers **only to a subject number** (not a unique patient identifier), classify it as **"Uncategorized"**.
2. Ensure **at least one column** is classified as **"Patient ID"**.

---

### **CRITICAL INSTRUCTIONS (DO NOT VIOLATE)**

* **DO NOT change column names in any way.**
* **DO NOT flatten, simplify, normalize, trim, or rewrite headers.**
* **DO NOT remove prefixes such as `"Unnamed: x_level_y"`**.
* **DO NOT remove units, superscripts, subscripts, or numeric suffixes**.
* **DO NOT merge multi-level headers into a single string**.
* **DO NOT infer or “clean up” column names.**

➡️ **The output key must be an EXACT character-for-character copy of the original column header**, including:

* tuple structure
* capitalization
* whitespace
* punctuation
* Unicode symbols (e.g., µ, −)

If a column header is represented as a tuple, **use the tuple verbatim as the key**.

---

### **Output Format**

Return **one JSON object** where:

* **Each key = one original column header copied exactly**
* **Each value = one category string**, chosen from:

  * `"Patient ID"`
  * `"Parameter value"`
  * `"Uncategorized"`

The **number of keys MUST exactly equal the number of columns**.
Do **not** add, remove, rename, or merge any keys.

---

### **Self-Check Before Finalizing**

Before producing the final answer, verify that:

* Every output key exactly matches one and only one input column header.
* No column name has been altered, shortened, or reformatted.
* The number of keys equals the number of input columns.

---

### **Example**

```json
{{
  "reasoning_process": "balahbalah",
  "categorized_headers": {{
    "('Unnamed: 0_level_0', 'Volunteer')": "Patient ID",
    "('Citalopram', 'Maximum1 milk concentration (µg l−1)')": "Parameter value",
    "('Unnamed: 4_level_0', 'M/PAUC')": "Parameter value"
  }}
}}
```

""")


def get_header_categorize_prompt(md_table_aligned: str):
    df_table = markdown_to_dataframe(md_table_aligned)
    processed_md_table_aligned = display_md_table(md_table_aligned)
    column_headers_str = "These are all its column headers: " + ", ".join(
        f'"{col}"' for col in df_table.columns
    )
    return HEADER_CATEGORIZE_PROMPT.format(
        processed_md_table_aligned=processed_md_table_aligned,
        column_headers_str=column_headers_str,
    )


class HeaderCategorizeResult(PKIndCommonAgentResult):
    """Categorized results for headers"""

    categorized_headers: dict[str, str] = Field(
        description="""the dictionary represents the categorized result for headers. Each key is a column header, and the corresponding value is its assigned category (one of the values: "Patient ID", "Parameter value", and "Uncategorized")"""
    )


## It seems it's LangChain's bug when trying to convert HeaderCategorizeResult into a JSON schema. It throws error:
## Error code 400 - Invalid schema for response_format 'HeaderCategorizeResult': In context=(), 'required' is required to be
##     supplied and to be an array including every key in properties. Extra required key 'categorized_headers' supplied.
## So, here we introduce json schema
HeaderCategorizeJsonSchema = {
    "title": "HeaderCategorizeResult",
    "description": "Categorized results for headers",
    "type": "object",
    "properties": {
        "reasoning_process": {
            "type": "string",
            "description": "A detailed explanation of the thought process or reasoning steps taken to reach a conclusion.",
            "title": "Reasoning Process",
        },
        "categorized_headers": {
            "type": "object",
            "description": 'the dictionary represents the categorized result for headers. Each key is a column header name, and the corresponding value is its assigned category string (one of the values: "Patient ID", "Parameter value", "Uncategorized")',
            "title": "Categorized Headers",
        },
    },
    "required": ["categorized_headers"],
}


def post_process_validate_categorized_result(
    result: HeaderCategorizeResult | dict,
    md_table_aligned: str,
) -> HeaderCategorizeResult:
    if isinstance(result, dict):
        try:
            res = HeaderCategorizeResult(**result)
        except ValidationError as e:
            logger.error(e)
            raise e
    else:
        res = result
    # Ensure column count matches the table
    expected_columns = markdown_to_dataframe(md_table_aligned).shape[1]
    match_dict = res.categorized_headers
    if len(match_dict.keys()) != expected_columns:
        error_msg = f"Mismatch: Expected {expected_columns} columns, but got {len(match_dict.keys())} in match_dict."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Ensure "Patient ID" column exists
    parameter_type_count = list(match_dict.values()).count("Patient ID")
    if parameter_type_count == 0:
        error_msg = f"**There must be at least one column that serves as the patient ID. Make sure you find it**"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return res
