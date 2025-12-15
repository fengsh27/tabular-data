from typing import Callable, Optional
from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd

from extractor.agents.common_agent.common_agent import RetryException
from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_pe_agents.pk_pe_common_step import PKPECommonStep
from extractor.agents.common_agent.common_agent import CommonAgent
from extractor.agents.common_agent.common_agent_2steps import CommonAgentTwoSteps
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState
from extractor.agents.pk_pe_agents.pk_pe_agents_utils import format_source_tables
from extractor.constants import COT_USER_INSTRUCTION
from extractor.request_openai import get_5_mini_openai
from extractor.request_geminiai import get_gemini
from extractor.request_gpt_oss import get_gpt_qwen_30b

PKPE_CORRECTION_SYSTEM_PROMPT = """
You are a biomedical data correction assistant with expertise in {domain}. 
You are given the **source paper title and tables**, and the **curated {domain} data table** that is incorrect, 
and the reasoning process of why the curated table is incorrect, which includes the **explanation** and **suggested fix**.

Your task is to carefully examine the **source paper title and tables**, and **reasoning process** of why the curated table is incorrect,
and then provide a corrected version of the curated table.

---

### **Your Responsibilities**

* fix the incorrect values or structure issues in the curated table.
* The final output should be a **complte and corrected version of the curated table**.

---

### **Input**

You will be given:

* **Paper Title**: The title of the publication.
* **Paper Abstract**: The abstract of the publication.
* **Source Table(s) or full text**: Table(s) extracted directly from the publication, preserving structure and labels, or the full text of the publication.
* **Curated Table**: The data table that has been curated from the above source for downstream use.
* **Reasoning Process**: The reasoning process of why the curated table is incorrect, which includes the **explanation** and **suggested fix**.
---

### **Your Output**

You must respond using the **exact compact json format** below:

```
{{
  "corrected_table":<A string, the corrected version of the curated table in markdown format>
}}
```

---

### **Important Instructions**

❗ You **must output the full corrected table**, even if only a few rows are changed.

* Treat the **Curated Table** as the **base**.
* Treat the **Suggested Fix** as a **delta or patch** to be applied to the base.
* **Only modify** the rows or values that are explicitly addressed in the **Reasoning Process**.
* **Preserve all other rows, columns, and formatting exactly as in the curated table**.
* Maintain the **original row order** unless the Reasoning Process explicitly requires adding, removing, or reordering rows.
* In your final answer, the corrected table must be in raw markdown format, **do not wrap the table in a code fence** (like '```' or '```markdown `).

---

### **Input**

#### **Paper Title**

{paper_title}

#### **Paper Abstract**

{paper_abstract}

#### **Source Table(s) or full text**

{source_tables}

#### **Curated Table**

```markdown
{curated_table}
```

#### **Reasoning Process**

{reasoning_process}

---

"""

class PKPECorrectionStepResult(BaseModel):
    corrected_table: str = Field(description="The corrected version of the curated table in markdown format.")

def post_process_corrected_table(res: PKPECorrectionStepResult) -> PKPECorrectionStepResult:
    """
    Post-process the corrected table to ensure it has the expected row number.
    """
    corrected_table = res.corrected_table = res.corrected_table.strip()
    try:
        df = markdown_to_dataframe(corrected_table)
        return res
    except Exception as e:
        raise RetryException(f"The corrected table is not a valid markdown table: {e}")

class PKPECuratedTablesCorrectionStep(PKPECommonStep):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        pmid: str,
        domain: str,
    ):
        super().__init__(llm)
        # if isinstance(llm, AzureChatOpenAI) and llm.model_name == "gpt-4o":
            # FIXME: gpt-4o does not work well to correct big tables, so we use gpt-5-mini instead.
            # self.llm = get_gemini()
            # self.llm = llm
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

        agent = self.get_agent(llm=self.llm) # CommonAgent(llm=self.llm)

        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Let's start the correction process.",# instruction_prompt,
            schema=PKPECorrectionStepResult,
        )
        self._print_step(state, step_output=reasoning_process if reasoning_process is not None else "N / A")
        res: PKPECorrectionStepResult = res
        self._print_step(state, step_output=f"Corrected Table: \n\n{res.corrected_table}")
        state["curated_table"] = res.corrected_table

        return state, token_usage

    def leave_step(self, state, token_usage: Optional[dict[str, int]] = None):
        return super().leave_step(state, token_usage)

