from typing import List
from pandas import DataFrame
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import logging

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.agent_utils import get_reasoning_process
from extractor.agents.common_agent.common_agent import CommonAgent, CommonAgentResult, RetryException
from extractor.agents.pk_summary.pk_sum_common_agent import (
    PKSumCommonAgentResult,
    # PKSumCommonAgent,
)
from extractor.agents.agent_factory import get_common_agent
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
  * Cord blood (umbilical venous, umbilical arterial)
  * Breast milk
  * Amniotic fluid
  * Cerebrospinal fluid (CSF)
  * Tissues

* **PK parameters (These are often abbreviated and will have associated units (e.g., ng/ml, L/h, hours))**, such as:

  * AUC (Area Under the Curve)
  * Cmax (Maximum concentration)
  * Tmax (Time to Cmax)
  * t½ (Half-life)
  * Elimination rate constant (Kel)
  * Volume of distribution 
  * Clearance (CL)
  * Ratios of concentration (e.g., breastmilk to plasma concentration ratio, mother to child serum concentration ratio), etc.

* **ADME-related characteristics**, including time profiles and cumulative excretion data.

---

### **Exclusion Criteria**

Do **not** include tables that:

* Primarily present **regression models**, **covariate analyses**, **population PK (PopPK) modeling results **, **statistical modeling**, or **correlational analyses between PK parameters and other variables**.
* Focus only on **patient demographics and baseline characteristics**, **treatment groups or study arm information**, or **non-PK safety outcomes**.

---

### **Input**

The content includes one or more tables in markdown format. Each table is preceded by a unique identifier like `"table_1"`, `"table_2"`, etc.
{table_content}
---

### **Your Output Format**

Return an **exact json compact format** below:

```json
{{
    "reasoning_process": "<reasoning process>",
    "selected_table_indexes": ["<table index>", "<table index>", ...]
}}
```

---

### **Output Example**

```json
{{
    "reasoning_process": "<reasoning process>",
    "selected_table_indexes": ["0", "1", ...]
}}
```

---

""")


class TablesSelectionResult(BaseModel):
    """Tables Selection Result"""
    reasoning_process: str = Field(description="A detailed explanation of the thought process or reasoning steps taken to reach a conclusion.")
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

    agent = get_common_agent(llm=llm) # PKSumCommonAgent(llm=llm)
    result = agent.go(
        system_prompt=system_prompt,
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=TablesSelectionResult,
        post_process=post_process_selected_table_ids,
        html_tables=html_tables,
    )
    res: TablesSelectionResult = result[0]
    tables = result[1]
    token_usage = result[2]
    reasoning_process = get_reasoning_process(result)
    logger.info(f"Selected tables (indices): {res.selected_table_indexes}")
    logger.info(f"Reason: {reasoning_process}")

    return tables, res.selected_table_indexes, reasoning_process, token_usage


SELECT_DEMOGRAPHIC_TABLES_PROMPT = ChatPromptTemplate.from_template("""
Analyze the provided content and identify all tables related to patient demographic data or baseline clinical characteristics. The goal is to capture the initial profile of the study population, not the results or outcomes of the study. These tables are often the first ones a publication and typically labeled with titles such as:
 · Patient Demographics
 · Baseline characteristics of the study population
 · Summary of patient characteristics
 · Demographic and clinical characteristics

---

### **Inclusion Criteria**

Focus particularly on tables that report Population-focused characteristics. Not PK parameter!!! 
    · “Age," “Sex," "Height or Length," "Weight," “Gender," “Race," “Ethnicity"
    · "Gravidity," "Parity," "Gestational Age," "Pre-existing medical conditions,"
    · "Birth Weight," "Apgar Score," "Head Circumference," "Feeding Status," 
    · “Socioeconomic status," “Education," “Marital status"
    · “Comorbidity," “Drug indication," “Adverse events"
    · “Severity," “BMI," “Smoking status," “Alcohol use," "Blood pressure"

---

### **Your Output Format**

Return a Python list of the relevant **table indexes** in the **exact format** below:

```python
[<table index>, <table index>, ...]
```

Do not include any explanations or extra output.

---

### **Output Example**

```python
["1", "3"]
```

---

### **Input**

The content including markdown table to analyze:
{table_content}

""")


def select_pk_demographic_tables(html_tables: list[dict[str, str | DataFrame]], llm):
    table_content = generate_tables_prompts(html_tables, True)
    system_prompt = SELECT_DEMOGRAPHIC_TABLES_PROMPT.format(table_content=table_content)

    agent = get_common_agent(llm=llm) # PKSumCommonAgent(llm=llm)
    res, tables, token_usage, reasoning_process = agent.go(
        system_prompt=system_prompt,
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=TablesSelectionResult,
        post_process=post_process_selected_table_ids,
        html_tables=html_tables,
    )

    if reasoning_process is None:
        reasoning_process = res.reasoning_process if hasattr(res, "reasoning_process") else "N / A"
    logger.info(f"Selected tables (indices): {res.selected_table_indexes}")
    logger.info(f"Reason: {reasoning_process}")

    return tables, res.selected_table_indexes, reasoning_process, token_usage


SELECT_PE_TABLES_PROMPT = ChatPromptTemplate.from_template("""
Analyze the provided content and identify all tables that include any clinical trial outcomes (both efficacy and safety), measurements potentially affected by the drug under study, or pharmaco-epidemiologic findings.

If a table contains any variables that may reflect:
 * Efficacy of the drug, such as treatment response, change from baseline, time-to-event 
 * Safety outcomes: counts or frequencies of adverse events, severity, serious adverse events, laboratory abnormalities, or other health indicators—include it.

If a table contains any variables that may focus on association, risk, and prevalence rather than the direct efficacy found in a controlled trial:
 * Measures of association, e.g., Odds ratio, Risk ratio or Relative risk, Hazard ratio
 * Incidence and prevalence - include it.


If such variables are present, the table should be included regardless of context or causality.

---

### **Your Output Format**

Return a Python list of the relevant **table indexes** in the **exact format** below:

```python
[<table index>, <table index>, ...]
```

---

### **Output Example**

```python
["1", "3"]
```

---

### **Input**

The content including markdown table to analyze:
{table_content}

""")


def select_pe_tables(html_tables: list[dict[str, str | DataFrame]], llm):
    table_content = generate_tables_prompts(html_tables, True)
    system_prompt = SELECT_PE_TABLES_PROMPT.format(table_content=table_content)

    agent = get_common_agent(llm=llm) # PKSumCommonAgent(llm=llm)
    res, tables, token_usage, reasoning_process = agent.go(
        system_prompt=system_prompt,
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=TablesSelectionResult,
        post_process=post_process_selected_table_ids,
        html_tables=html_tables,
    )

    logger.info(f"Selected tables (indices): {res.selected_table_indexes}")
    logger.info(f"Reason: {reasoning_process}")

    return tables, res.selected_table_indexes, reasoning_process, token_usage

def add_row_index_column(md_table: str) -> str:
    df = markdown_to_dataframe(md_table)
    df.insert(0, "rowIndex", range(len(df)))
    return dataframe_to_markdown(df)
    