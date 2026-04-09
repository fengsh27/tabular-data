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
  "explanation": <string, max 200 words. If incorrect, list ALL errors using idx (0-based positional index) in the compact format below. If correct, state "All values match." and nothing more.>,
  "suggested_fix": <string or None, if incorrect, repeat each error with the fix using idx (0-based positional index) and the compact format below.>
}}
```

---

### **Error Listing Format (MUST FOLLOW when incorrect)**

Identify each cell by its **0-based positional index** in the curated table (first data row after the header = idx 0, second data row = idx 1, etc.).

**CRITICAL**: The index is the row's POSITION in the table, NOT a value from any column. For example, if the first data row has Patient ID "3", that row is still **idx 0** (not idx 3).

List EVERY error as one line each, using this compact format:
  idx 0, Col "column_name": change "found_value" to "expected_value"

For missing rows:
  Missing row: Col1="val1", Col2="val2", ...

For extra rows that should be removed:
  Extra idx X: should be removed

---

### **Important Rules**

* The columns in the curated table are fixed — do NOT question column names or order.
* Focus on **substantial mismatches** in values or structure. Minor typos, slight wording differences, or small formatting variations are acceptable. 
  In particular, ignore footnote markers or superscripts attached to values (e.g., "< LODa" vs "< LOD", "3.5*" vs "3.5").
* When values in text and table disagree, treat the **table values as ground truth**.
* You MUST list **EVERY** incorrect value. Do NOT use phrases like "for instance", "for example", "such as", "e.g.", or "etc." to give partial examples. An incomplete error list means corrections will be incomplete.
* Do NOT explain WHY a value is wrong — just state WHAT is wrong and WHAT it should be.
* Keep explanation under 200 words. No reasoning, no justification — only the error list.

---

### **Output Example**

```
{{
  "correct": false,
  "explanation": "idx 3, Col \"Parameter value\": change \"4.13\" to \"< LOD\"\nidx 20, Col \"Parameter value\": change \"37\" to \"0.37\"",
  "suggested_fix": "idx 3, Col \"Parameter value\": change \"4.13\" to \"< LOD\"\nidx 20, Col \"Parameter value\": change \"37\" to \"0.37\""
}}
```

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

### **Previous Verification Attempts** (if any)

{previous_verification_thoughts}

If this section is not empty, use it to understand what was already checked and whether prior corrections resolved those issues. Do NOT re-report issues that have already been fully resolved.

---

"""

class PKPEVerificationStepResult(BaseModel):
    # reasoning_process: str = Field(description="A **concise explanation** of the thought process or reasoning steps taken to reach a conclusion (no more than 200 words).")
    correct: bool = Field(description="Whether the curated table is accurate and faithful to the source table(s).")
    explanation: str = Field(description="If incorrect, list ALL errors in compact format: idx X, Col Y: change found to expected. Max 200 words. If correct, state 'All values match.'")
    suggested_fix: Optional[str] = Field(default=None, description="If incorrect, repeat each error: idx X, Col \"name\": change \"wrong_value\" to \"correct_value\"")
    
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
            prev_thoughts = state.get("previous_verification_thoughts") or []
            prev_thoughts_str = "\n\n".join(
                f"Attempt {i+1}:\n{t}" for i, t in enumerate(prev_thoughts)
            ) if prev_thoughts else "None"

            system_prompt = PKPE_VERIFICATION_SYSTEM_PROMPT.format(
                paper_title=state["paper_title"],
                paper_abstract=state["paper_abstract"],
                source_tables=source_tables,
                curated_table=state["curated_table"],
                domain=self.domain,
                previous_verification_thoughts=prev_thoughts_str,
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
            entry = f"Explanation: {state['explanation']}\nSuggested fix: {state['suggested_fix']}"
            thoughts = state.get("previous_verification_thoughts") or []
            thoughts.append(entry)
            state["previous_verification_thoughts"] = thoughts[-2:]
        valid_reasoning = reasoning_process if isinstance(reasoning_process, str) and reasoning_process.strip() else None
        state["verification_reasoning_process"] = valid_reasoning or state["suggested_fix"] or state["explanation"]

        return state, token_usage

    def leave_step(self, state, token_usage: Optional[dict[str, int]] = None):
        return super().leave_step(state, token_usage)

