from typing import List
from pandas import DataFrame
from pydantic import Field
from langchain_core.prompts import ChatPromptTemplate
import logging

from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.common_agent.common_agent import CommonAgent, CommonAgentResult, RetryException
from extractor.agents.pk_summary.pk_sum_common_agent import (
    PKSumCommonAgentResult,
    PKSumCommonAgent,
)
from extractor.prompts_utils import generate_tables_prompts

logger = logging.getLogger(__name__)

SELECT_PK_TABLES_PROMPT = ChatPromptTemplate.from_template("""
You are a biomedical data assistant specializing in pharmacokinetics (PK). 
Your task is to carefully analyze the provided content and **identify all tables relevant to pharmacokinetics (PK)**, 
specifically those related to **ADME properties** (Absorption, Distribution, Metabolism, and Excretion).

---

### **Inclusion Criteria**

Select tables that include any of the following:

* **Drug concentration measurements** in biological matrices such as:

  * Plasma
  * Serum
  * Urine
  * Cord blood
  * Tissues

* **PK parameters**, such as:

  * AUC (Area Under the Curve)
  * Cmax (Maximum concentration)
  * Tmax (Time to Cmax)
  * t½ (Half-life)
  * Volume of distribution, clearance, bioavailability, etc.

* **ADME-related characteristics**, including time profiles and cumulative excretion data.

---

### **Exclusion Criteria**

Do **not** include tables that:

* Primarily present **regression models**, **statistical modeling**, or **correlational analyses** of PK parameters.
* Focus only on **demographics**, **treatment groups**, or **non-PK safety outcomes**.

---

### **Input**

The content includes one or more tables in markdown format. Each table is preceded by a unique identifier like `"table_1"`, `"table_2"`, etc.
{table_content}
---

### **Your Output Format**

Return a Python list of the relevant **table indexes** in the **exact format** below:

```python
["<table_index_1>", "<  table_index_2>", ...]
```

Do not include any explanations or extra output.

---

### **Output Example**

```python
["1", "3"]
```

---

""")


class TablesSelectionResult(PKSumCommonAgentResult):
    """Tables Selection Result"""

    selected_table_indexes: List[str] = Field(
        description="""a list of selected table indexes, such as ["1", "2", "3"]"""
    )


def post_process_selected_table_ids(
    res: TablesSelectionResult,
    html_tables: list[dict[str, str | DataFrame]],
):
    ids = res.selected_table_indexes
    if ids is None:
        raise ValueError("Invalid selected tables")

    indices = []
    for id in ids:
        try:
            if id.startswith('"') and id.endswith('"') and len(id) > 2:
                id = id[1:-1]
            id = int(id)
        except ValueError:
            raise RetryException(
                f"""Please generate valid table id and **exactly follow the format**: ["table_index_1", "table_index_2", ...]. \n\nWrong answer example: `{id}`"""
            )

        if id < 0 or id > len(html_tables):
            raise RetryException(
                "Please generate valid table id, wrong answer example: `{id}`"
            )
        indices.append(id)

    tables = []
    for ix in indices:
        tables.append(html_tables[ix])

    return tables


def select_pk_summary_tables(html_tables: list[dict[str, str | DataFrame]], llm):
    table_content = generate_tables_prompts(html_tables, True)
    system_prompt = SELECT_PK_TABLES_PROMPT.format(table_content=table_content)

    agent = PKSumCommonAgent(llm=llm)
    res, tables, token_usage = agent.go(
        system_prompt=system_prompt,
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=TablesSelectionResult,
        post_process=post_process_selected_table_ids,
        html_tables=html_tables,
    )

    logger.info(f"Selected tables (indices): {res.selected_table_indexes}")
    logger.info(f"Reason: {res.reasoning_process}")

    return tables, res.selected_table_indexes, res.reasoning_process, token_usage


SELECT_DEMOGRAPHIC_TABLES_PROMPT = ChatPromptTemplate.from_template("""
Analyze the provided content and identify all tables related to demographic data. 

Focus particularly on tables that report Population-focused characteristics. Not PK parameter!!!
    · “Age," “Sex," "Weight," “Gender," “Race," “Ethnicity"
    · “Socioeconomic status," “Education," “Marital status"
    · “Comorbidity," “Drug indication," “Adverse events"
    · “Severity," “BMI," “Smoking status," “Alcohol use," "Blood pressure"

Return the results as a Python list of table indexes in this exact format:
["table_index_1", "table_index_2", ...]

The content including markdown table to analyze:
{table_content}
""")


def select_pk_demographic_tables(html_tables: list[dict[str, str | DataFrame]], llm):
    table_content = generate_tables_prompts(html_tables, True)
    system_prompt = SELECT_DEMOGRAPHIC_TABLES_PROMPT.format(table_content=table_content)

    agent = PKSumCommonAgent(llm=llm)
    res, tables, token_usage = agent.go(
        system_prompt=system_prompt,
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=TablesSelectionResult,
        post_process=post_process_selected_table_ids,
        html_tables=html_tables,
    )

    logger.info(f"Selected tables (indices): {res.selected_table_indexes}")
    logger.info(f"Reason: {res.reasoning_process}")

    return tables, res.selected_table_indexes, res.reasoning_process, token_usage


SELECT_PE_TABLES_PROMPT = ChatPromptTemplate.from_template("""
Analyze the provided content and identify all tables that include any outcomes or measurements potentially affected by the drug under study.

If a table contains any variables that may reflect the effect of drug exposure—such as clinical outcomes, adverse events, treatment response, laboratory values, or other health indicators—include it.

If such variables are present, the table should be included regardless of context or causality.

Return the results as a Python list of table indexes in the following format:
["table_index_1", "table_index_2", ...]

The content, including markdown tables, to analyze:
{table_content}
""")


def select_pe_tables(html_tables: list[dict[str, str | DataFrame]], llm):
    table_content = generate_tables_prompts(html_tables, True)
    system_prompt = SELECT_PE_TABLES_PROMPT.format(table_content=table_content)

    agent = PKSumCommonAgent(llm=llm)
    res, tables, token_usage = agent.go(
        system_prompt=system_prompt,
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=TablesSelectionResult,
        post_process=post_process_selected_table_ids,
        html_tables=html_tables,
    )

    logger.info(f"Selected tables (indices): {res.selected_table_indexes}")
    logger.info(f"Reason: {res.reasoning_process}")

    return tables, res.selected_table_indexes, res.reasoning_process, token_usage