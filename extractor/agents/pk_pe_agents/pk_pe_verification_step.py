from typing import Callable, Optional
import logging
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE
from extractor.agents.common_agent.common_agent import CommonAgent
from extractor.agents.pk_pe_agents.pk_pe_common_step import PKPECommonStep
from extractor.agents.common_agent.common_agent_2steps import CommonAgentTwoSteps
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState, FinalAnswerEnum
from extractor.agents.pk_pe_agents.pk_pe_agents_utils import format_source_tables
from extractor.constants import COT_USER_INSTRUCTION
from extractor.request_gpt_oss import get_gpt_qwen_30b

logger = logging.getLogger(__name__)

PKPE_VERIFICATION_SYSTEM_PROMPT = """
You are a biomedical data verification assistant with expertise in {domain} and data accuracy validation. 
Your task is to carefully examine the **source paper title and tables**, and determine whether the **curated {domain} data table** is an accurate and faithful representation of the information provided in the source.

---

### **Your Responsibilities**

* Verify that all values in the curated table exactly match or are correctly derived from the source table(s) in the paper.
* Check that the table structure (rows, columns, units, and headers) is curated correctly from the source.
* Identify any discrepancies in numerical values, missing data, wrong units, or incorrect associations (e.g., a value placed in the wrong row or column).
* Consider the context from the paper title if needed (e.g., study type, drug, population) to interpret ambiguous values.

---

### **Input**

You will be given:

* **Paper Title**: The title of the publication.
* **Paper Abstract**: The abstract of the publication.
* **Source Table(s) or full text**: Table(s) extracted directly from the publication, preserving structure and labels, or the full text of the publication.
* **Curated Table**: The data table that has been curated from the above source for downstream use.

---

### **Your Output**

You must respond using the **exact json compact format** below:

```
{{
  "correct": <boolean, True / False>,
  "explanation": <string, brief explanation of whether the curated table is accurate. If incorrect, explain what is wrong, including specific mismatched values or structure issues>,
  "suggested_fix": <string or None, if incorrect, provide a corrected version of the curated table or the corrected values/rows/columns.>
}}
```

---

### **Important Notes**

* The columns in the curated table are fixed, so you **should not doubt** the columns in the curated table.
* Focus on **substantial mismatches** in values or structure that could affect the meaning or interpretation. Minor typos, slight wording differences, or small formatting variations are acceptable.
* If the curated table is correct in content but uses slightly different formatting (e.g., reordering of columns), that is acceptable as long as it does not alter the meaning or value.
* In the **Explanation** section, you should try your best to **list all the mismatched values or structure issues**, and provide a brief explanation of why you think the curated table is incorrect.
* Your response will be used to correct the curated table, so you should be **very specific and detailed** in your explanation. **Do not give any general explanation.**
* when values in text and table disagree, treat the table values as the ground truth (even if the text mentions slightly different ones).
---

### **Input**

#### **Paper Title**

{paper_title}

#### **Paper Abstract**

{paper_abstract}

#### **Source Table(s) or full text**

{source_tables}

#### **Curated Table**

{curated_table}

---

"""

class PKPEVerificationStepResult(BaseModel):
    # reasoning_process: str = Field(description="A **concise explanation** of the thought process or reasoning steps taken to reach a conclusion (no more than 200 words).")
    correct: bool = Field(description="Whether the curated table is accurate and faithful to the source table(s).")
    explanation: str = Field(description="Brief explanation of whether the curated table is accurate. If incorrect, explain what is wrong, including specific mismatched values or structure issues.")
    suggested_fix: Optional[str] = Field(description="If incorrect, provide a corrected version of the curated table or the corrected values/rows/columns.")
    
class PKPECuratedTablesVerificationStep(PKPECommonStep):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        pmid: str,
        domain: str,
    ):
        super().__init__(llm)
        self.step_name = "PK PE Verification Step"
        self.pmid = pmid
        self.domain = domain

    def _update_intermediate_output(self, state, explanation, suggested_fix):
        error_msg = """
        #### **Error**
Explanation: 
{explanation}

Suggested fix: 
{suggested_fix}

"""
        if not "previous_errors" in state or state["previous_errors"] is None:
            state["previous_errors"] = error_msg
        else:
            state["previous_errors"] += f"\n\n{error_msg}"

    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        state: PKPECurationWorkflowState = state

        # If a previous step already set a terminal non-correctable answer, skip.
        answer = state.get("final_answer")
        if answer is not None and answer.is_terminal:
            return state, {**DEFAULT_TOKEN_USAGE}

        source_tables = state["source_tables"] if "source_tables" in state else None
        source_tables = format_source_tables(source_tables)
        raw = state.get("curated_table")
        curated_table = raw.strip() if isinstance(raw, str) else None
        curated_table = curated_table if curated_table else None
        if curated_table is None:
            state["final_answer"] = FinalAnswerEnum.NoTable
            state["explanation"] = "No data was curated from the source."
            state["suggested_fix"] = "N/A"
            return state, {**DEFAULT_TOKEN_USAGE}

        try:
            system_prompt = PKPE_VERIFICATION_SYSTEM_PROMPT.format(
                paper_title=state["paper_title"],
                paper_abstract=state["paper_abstract"],
                source_tables=source_tables,
                curated_table=state["curated_table"],
                domain=self.domain,
            )
            instruction_prompt = COT_USER_INSTRUCTION
            agent = self.get_agent(llm=self.llm)
            res, _, token_usage, reasoning_process = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=PKPEVerificationStepResult,
            )
        except Exception as e:
            logger.error(f"Error running verification agent: {e}")
            state["final_answer"] = FinalAnswerEnum.VerificationError
            state["explanation"] = f"Error running verification agent: {e}"
            state["suggested_fix"] = "N/A"
            return state, {**DEFAULT_TOKEN_USAGE}

        if reasoning_process is None:
            reasoning_process = res.reasoning_process if hasattr(res, "reasoning_process") else None
        self._print_step(state, step_output=reasoning_process or "N / A")
        self._print_step(state, step_output=f"Verification Final Answer: \n\n{res.correct}")
        self._print_step(state, step_output=f"Verification Explanation: \n\n{res.explanation}")
        self._print_step(state, step_output=f"Verification Suggested Fix: \n\n{res.suggested_fix}")
        state["final_answer"] = FinalAnswerEnum.Correct if res.correct else FinalAnswerEnum.Incorrect
        state["explanation"] = res.explanation
        suggested_fix = res.suggested_fix if isinstance(res.suggested_fix, str) and res.suggested_fix.strip() else None
        state["suggested_fix"] = suggested_fix if suggested_fix is not None else res.explanation

        if not res.correct:
            self._update_intermediate_output(state, state["explanation"], state["suggested_fix"])
        valid_reasoning = reasoning_process if isinstance(reasoning_process, str) and reasoning_process.strip() else None
        state["verification_reasoning_process"] = valid_reasoning or state["suggested_fix"] or state["explanation"]

        return state, token_usage

    def leave_step(self, state, token_usage: Optional[dict[str, int]] = None):
        return super().leave_step(state, token_usage)

