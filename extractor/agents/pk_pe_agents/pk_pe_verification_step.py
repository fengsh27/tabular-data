from typing import Callable, Optional
import logging
import re
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
  In particular, 
    - **ignore** footnote markers or superscripts attached to values (e.g., "< LODa" vs "< LOD", "3.5*" vs "3.5").
    - **ignore** the difference between "nan" and "N/A" and "N/A" and "NA", and so on.
* In explanation, **must not** include thinking process, such as "re-evaluate", "wait", "re-check", "Let's check", etc, and **must not** exceed 200 words.
* When values in text and table disagree, treat the **table values as ground truth**.
* You MUST list **EVERY** incorrect value. Do NOT use phrases like "for instance", "for example", "such as", "e.g.", or "etc." to give partial examples. An incomplete error list means corrections will be incomplete.
* Do NOT explain WHY a value is wrong — just state WHAT is wrong and WHAT it should be.
* Only list rows that have ACTUAL errors. Do NOT list rows where the current value already matches the expected value (e.g., do NOT write: idx 2, Col "P value": change "0.001" to "0.001").
* **Keep explanation under 200 words**. No reasoning, no justification — only the error list.

---

### **Output Example**

CORRECT — only list actual errors:
```
{{
  "correct": false,
  "explanation": "idx 3, Col \"Parameter value\": change \"4.13\" to \"< LOD\"\nidx 20, Col \"Parameter value\": change \"37\" to \"0.37\"",
  "suggested_fix": "idx 3, Col \"Parameter value\": change \"4.13\" to \"< LOD\"\nidx 20, Col \"Parameter value\": change \"37\" to \"0.37\""
}}
```

WRONG cases — do NOT list correct rows or no-op changes:
```
{{
  "correct": false,
  "explanation": "idx 0, Col \"P value\": change \"0.01\" to \"0.01\"\nidx 1, Col \"P value\": change \"N/A\" to \"0.007\"\nidx 2, Col \"P value\": change \"0.04\" to \"0.04\"",
  "suggested_fix": "idx 0, Col \"P value\": change \"0.01\" to \"0.01\"\nidx 1, Col \"P value\": change \"N/A\" to \"0.007\"\nidx 2, Col \"P value\": change \"0.04\" to \"0.04\""
}}
```
Above is WRONG because idx 0 and idx 2 have no actual change ("0.01" to "0.01" and "0.04" to "0.04"). The correct output should only include idx 1:
```
{{
  "correct": false,
  "explanation": "idx 1, Col \"P value\": change \"N/A\" to \"0.007\"",
  "suggested_fix": "idx 1, Col \"P value\": change \"N/A\" to \"0.007\""
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

If this section is not empty:
- Use it to understand what was already checked and whether prior corrections resolved those issues.
- Do NOT re-report issues that have already been fully resolved.
- **CRITICAL: Do NOT revert a previous correction.** If a previous attempt changed a value from A to B, and the current table now has B, do NOT suggest changing it back to A. A correction that was already applied is considered resolved — accept the current value.

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

    _CHANGE_PATTERN = re.compile(r'change\s+"([^"]*)"\s+to\s+"([^"]*)"')
    _IDX_COL_PATTERN = re.compile(r'(idx\s+\d+,\s*Col\s+"[^"]*")')
    _IDX_NUM_PATTERN = re.compile(r'idx\s+(\d+)')
    _COL_NAME_PATTERN = re.compile(r'Col\s+"([^"]*)"')

    @staticmethod
    def _remove_noop_fixes(text: str) -> str:
        """Remove lines where the 'change' from/to values are identical, e.g.
        idx 2, Col "P value": change "0.001" to "0.001"
        """
        if not text:
            return text
        lines = text.split("\n")
        filtered = []
        for line in lines:
            m = PKPECuratedTablesVerificationStep._CHANGE_PATTERN.search(line)
            if m and m.group(1) == m.group(2):
                continue
            filtered.append(line)
        return "\n".join(filtered).strip()

    @staticmethod
    def _remove_oscillation_fixes(text: str, previous_thoughts: list[str]) -> str:
        """Remove lines that revert a previous fix (oscillation detection).
        E.g., previous said: idx 1, Col "X": change "A" to "B"
             current says:  idx 1, Col "X": change "B" to "A"
        """
        if not text or not previous_thoughts:
            return text
        # Build a set of (cell_key, from, to) from previous thoughts
        prev_fixes = set()
        for thought in previous_thoughts:
            for line in thought.split("\n"):
                idx_m = PKPECuratedTablesVerificationStep._IDX_COL_PATTERN.search(line)
                change_m = PKPECuratedTablesVerificationStep._CHANGE_PATTERN.search(line)
                if idx_m and change_m:
                    cell_key = idx_m.group(1).lower().replace(" ", "")
                    prev_fixes.add((cell_key, change_m.group(1), change_m.group(2)))

        if not prev_fixes:
            return text

        lines = text.split("\n")
        filtered = []
        for line in lines:
            idx_m = PKPECuratedTablesVerificationStep._IDX_COL_PATTERN.search(line)
            change_m = PKPECuratedTablesVerificationStep._CHANGE_PATTERN.search(line)
            if idx_m and change_m:
                cell_key = idx_m.group(1).lower().replace(" ", "")
                # Current wants to change B→A, but previous changed A→B — this is a revert
                if (cell_key, change_m.group(2), change_m.group(1)) in prev_fixes:
                    logger.info(f"Oscillation detected, removing: {line.strip()}")
                    continue
            filtered.append(line)
        return "\n".join(filtered).strip()

    @staticmethod
    def _remove_stale_fixes(text: str, curated_table: str) -> str:
        """Remove lines where the 'from' value doesn't match the actual value in the curated table.
        E.g., suggested fix says idx 52, Col "Parameter value": change "52" to "55.1"
        but the actual value at idx 52 is already "55.1" — this fix is stale/wrong.
        """
        if not text or not curated_table:
            return text
        try:
            from TabFuncFlow.utils.table_utils import markdown_to_dataframe
            df = markdown_to_dataframe(curated_table)
        except Exception:
            return text

        lines = text.split("\n")
        filtered = []
        for line in lines:
            idx_m = PKPECuratedTablesVerificationStep._IDX_NUM_PATTERN.search(line)
            col_m = PKPECuratedTablesVerificationStep._COL_NAME_PATTERN.search(line)
            change_m = PKPECuratedTablesVerificationStep._CHANGE_PATTERN.search(line)
            if idx_m and col_m and change_m:
                idx = int(idx_m.group(1))
                col_name = col_m.group(1)
                from_value = change_m.group(1)
                if col_name in df.columns and idx < len(df):
                    actual_value = str(df.iloc[idx][col_name]).strip()
                    if actual_value != from_value:
                        logger.info(
                            f"Stale fix removed: {line.strip()} "
                            f"(actual value is \"{actual_value}\", not \"{from_value}\")"
                        )
                        continue
            filtered.append(line)
        return "\n".join(filtered).strip()

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
        # Filter out no-op fixes (e.g., change "0.001" to "0.001")
        filtered_explanation = self._remove_noop_fixes(res.explanation)
        filtered_suggested_fix = self._remove_noop_fixes(res.suggested_fix) if res.suggested_fix else None

        # Filter out oscillation fixes (reverting a previous correction)
        prev_thoughts = state.get("previous_verification_thoughts") or []
        if prev_thoughts:
            filtered_explanation = self._remove_oscillation_fixes(filtered_explanation, prev_thoughts)
            if filtered_suggested_fix:
                filtered_suggested_fix = self._remove_oscillation_fixes(filtered_suggested_fix, prev_thoughts)

        # Filter out stale fixes (where the "from" value doesn't match the actual table value)
        curated_table = state.get("curated_table")
        if curated_table:
            filtered_explanation = self._remove_stale_fixes(filtered_explanation, curated_table)
            if filtered_suggested_fix:
                filtered_suggested_fix = self._remove_stale_fixes(filtered_suggested_fix, curated_table)

        self._print_step(state, step_output=f"Verification Explanation: \n\n{filtered_explanation}")
        self._print_step(state, step_output=f"Verification Suggested Fix: \n\n{filtered_suggested_fix}")
        # If all fixes were no-ops, oscillations, or stale, treat as correct
        if not res.correct and (filtered_suggested_fix is None or not filtered_suggested_fix.strip()):
            logger.info("All suggested fixes were no-ops, oscillations, or stale after filtering; treating as correct.")
            self._print_step(state, step_output="All suggested fixes were no-ops, oscillations, or stale; treating as correct.")
            state["final_answer"] = FinalAnswerEnum.Correct
            state["explanation"] = "All values match after filtering."
            state["suggested_fix"] = "N/A"
            return state, token_usage

        state["final_answer"] = FinalAnswerEnum.Correct if res.correct else FinalAnswerEnum.Incorrect
        state["explanation"] = filtered_explanation
        suggested_fix = filtered_suggested_fix if isinstance(filtered_suggested_fix, str) and filtered_suggested_fix.strip() else None
        state["suggested_fix"] = suggested_fix if suggested_fix is not None else filtered_explanation

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

