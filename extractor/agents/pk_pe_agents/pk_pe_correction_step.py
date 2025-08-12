from typing import Callable, Optional
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from extractor.agents.common_agent.common_step import CommonStep
from extractor.agents.common_agent.common_agent_2steps import CommonAgentTwoSteps
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState
from extractor.agents.pk_pe_agents.pk_pe_agents_utils import format_source_tables
from extractor.constants import COT_USER_INSTRUCTION

PKPE_CORRECTION_SYSTEM_PROMPT = """
You are a biomedical data verification assistant with expertise in {domain} and data accuracy validation. 
You are given the **source paper title and tables**, and the **curated {domain} data table** that is incorrect, 
and the reasoning process of why the curated table is incorrect, which includes the **explanation** and **suggested fix**.

Your task is to carefully examine the **source paper title and tables**, and **reasoning process** of why the curated table is incorrect,
and provide a corrected version of the curated table.

---

### **Your Responsibilities**

* Only fix the incorrect values or structure issues in the curated table, do not change other parts of the curated table.
* The final output should be a **complte and corrected version of the curated table**.

---

### **Input**

You will be given:

* **Paper Title**: The title of the publication.
* **Paper Abstract**: The abstract of the publication.
* **Source Table(s)**: Table(s) extracted directly from the publication, preserving structure and labels.
* **Curated Table**: The data table that has been curated from the above source for downstream use.
* **Reasoning Process**: The reasoning process of why the curated table is incorrect, which includes the **explanation** and **suggested fix**.

---

### **Your Output**

You must respond using the **exact format** below:

```
**FinalAnswer**: [The corrected version of the curated table in markdown format]
```

---

### **Important Notes**

* Only fix the incorrect values or structure issues in the curated table, do not change other parts of the curated table.
* Only make changes according to the **SuggestedFix** in the **Reasoning Process**.
* Note: The **SuggestedFix** in the **Reasoning Process** only is the changes that you need to make to the curated table, **do keep the other parts of the curated table unchanged**.
  So you final answer should be a **complete and corrected** version of the curated table, including the **SuggestedFix** and the unchanged parts of the curated table.

---

### **Input**

#### **Paper Title**

{paper_title}

#### **Paper Abstract**

{paper_abstract}

#### **Source Table(s)**

{source_tables}

#### **Curated Table**

{curated_table}

#### **Reasoning Process**

{reasoning_process}

---

"""

class PKPECorrectionStepResult(BaseModel):
    corrected_table: str = Field(description="The corrected version of the curated table in markdown format.")

class PKPECuratedTablesCorrectionStep(CommonStep):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        pmid: str,
        domain: str,
    ):
        super().__init__(llm)
        self.step_name = "PK PE Correction Step"
        self.pmid = pmid
        self.domain = domain

    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        state: PKPECurationWorkflowState = state
        source_tables = state["source_tables"] if "source_tables" in state else None
        source_tables = format_source_tables(source_tables)
        verification_reasoning_process = state["verification_reasoning_process"] if "verification_reasoning_process" in state else "N / A"
        system_prompt = PKPE_CORRECTION_SYSTEM_PROMPT.format(
            paper_title=state["paper_title"],
            paper_abstract=state["paper_abstract"],
            source_tables=source_tables,
            curated_table=state["curated_table"],
            reasoning_process=verification_reasoning_process,
            domain=self.domain,
        )
        instruction_prompt = COT_USER_INSTRUCTION

        agent = CommonAgentTwoSteps(llm=self.llm)

        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            schema=PKPECorrectionStepResult,
        )
        self._print_step(state, step_output=reasoning_process)
        res: PKPECorrectionStepResult = res
        self._print_step(state, step_output=f"Corrected Table: \n\n{res.corrected_table}")
        state["curated_table"] = res.corrected_table

        return state, token_usage

    def leave_step(self, state, token_usage: Optional[dict[str, int]] = None):
        return super().leave_step(state, token_usage)

