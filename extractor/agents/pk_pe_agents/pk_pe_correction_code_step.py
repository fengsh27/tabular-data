from typing import Callable, Optional
from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd
import logging

from extractor.agents.common_agent.common_agent import RetryException
from TabFuncFlow.utils.table_utils import markdown_to_dataframe, dataframe_to_markdown
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table, increase_token_usage
from extractor.agents.pk_pe_agents.pk_pe_common_step import PKPECommonStep
from extractor.agents.common_agent.common_agent import CommonAgent
from extractor.agents.common_agent.common_agent_2steps import CommonAgentTwoSteps
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState, FinalAnswerEnum
from extractor.agents.pk_pe_agents.pk_pe_agents_utils import format_source_tables
from extractor.agents.custom_python_ast_repl_tool import CustomPythonAstREPLTool
from extractor.constants import COT_USER_INSTRUCTION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1 — Planning prompt: decompose corrections into small tasks
# ---------------------------------------------------------------------------

PKPE_CORRECTION_PLANNING_PROMPT = '''
You are a biomedical data correction engineer with expertise in {domain}.

You are given a curated table and a Reasoning Process that describes what is wrong and what to fix.
Your job is to decompose the corrections into a list of **small, independent tasks**.

Each task should describe ONE logical group of edits that can be applied in a single short Python code snippet (under 30 lines).

Rules:
- Each task must be a concise, self-contained description of what to change.
- Group related edits together (e.g., "Fix Subject N for Adolescents 12-17: Day 3=7, Day 30=2" is one task).
- Do NOT group unrelated edits into one task.
- Do NOT write any code — only write task descriptions.
- Include ALL corrections from the Reasoning Process; do not skip any.
- IGNORE no-op entries where the value is unchanged (e.g., idx 2, Col "Parameter value": change "3" to "3"). Do NOT generate a task for these.
- IMPORTANT: "idx" in the Reasoning Process refers to 0-based row index (idx 0 = first data row of the DataFrame, i.e. df.iloc[0]). Do NOT treat it as 1-based.

--------------------
Paper Title:
{paper_title}

Source Table(s) or full text:
{source_tables}

Curated Table:
{curated_table}

Reasoning Process:
{reasoning_process}
--------------------

'''

# ---------------------------------------------------------------------------
# Step 2 — Per-task code generation prompt
# ---------------------------------------------------------------------------

PKPE_CORRECTION_TASK_PROMPT = '''
You are a biomedical data correction engineer with expertise in {domain} and robust Python data wrangling.

You are given a curated table and ONE specific correction task. Write Python code to apply ONLY this task.

CRITICAL: markdown_to_dataframe() is already implemented and available at runtime.
- You MUST NOT define it, re-implement it, or include any parsing logic that duplicates it.
- Call it directly as: df = markdown_to_dataframe(curated_md)

Your output must be valid JSON matching EXACTLY this schema:
  {{"code": "<python code as a single string WITHOUT code fences>"}}

Input/Output contract:
- curated_md (str) will be provided at runtime.
- You must produce df_corrected (pandas.DataFrame).
- df_corrected must preserve the same columns (names and order) as the curated table header.
- All cell values must remain strings unless the task explicitly requires type conversion.

IMPORTANT: "idx" in the correction task refers to 0-based row index (idx 0 = first data row of the DataFrame, i.e. df.iloc[0]). Do NOT treat it as 1-based.

When the task specifies idx numbers, use df.at[idx, "column_name"] to target cells directly. Do NOT use value-based matching (e.g., df.loc[df["col"] == "value"]) to locate rows — the values may be ambiguous or duplicated. Always use the idx provided.

Required structure of the code (enforced order):
1) import pandas as pd
2) df = markdown_to_dataframe(curated_md)
3) Apply the edits for this task ONLY
4) df_corrected = df

CRITICAL: The variable df_corrected MUST always be assigned. If no corrections are needed, still write: df_corrected = df

DO NOT:
- Define markdown_to_dataframe
- Re-parse markdown manually
- Change column names
- Reorder rows unless explicitly required
- Write comments or docstrings in the code
- Build new_rows lists or pd.DataFrame([...]) with hardcoded rows
- Keep the code under 30 lines

--------------------
Paper Title:
{paper_title}

Source Table(s) or full text:
{source_tables}

Curated Table:
(curated markdown table string will be assigned to curated_md at runtime)
{curated_table}

Correction Task:
{task_description}
--------------------

'''

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PKPECorrectionPlanResult(BaseModel):
    tasks: list[str] = Field(description="A list of correction task descriptions, each describing one logical group of edits.")

class PKPECorrectionStepResult(BaseModel):
    code: str = Field(description="Python code that corrects the curated table. The code must produce a pandas DataFrame named `df_corrected`.")

# ---------------------------------------------------------------------------
# Correction code step
# ---------------------------------------------------------------------------

MAX_CODE_RETRIES = 3

class PKPECuratedTablesCorrectionCodeStep(PKPECommonStep):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        pmid: str,
        domain: str,
    ):
        super().__init__(llm)
        self.step_name = "PK PE Correction Code Step"
        self.pmid = pmid
        self.domain = domain

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        state: PKPECurationWorkflowState = state
        source_tables = state["source_tables"] if "source_tables" in state else None
        source_tables = format_source_tables(source_tables)
        verification_reasoning_process = state["verification_reasoning_process"] if "verification_reasoning_process" in state else "N / A"
        curated_md = state["curated_table"]

        total_token_usage = {**DEFAULT_TOKEN_USAGE}

        # Step 1: Plan — decompose into tasks
        tasks, token_usage = self._plan_corrections(state, source_tables, verification_reasoning_process, curated_md)
        total_token_usage = increase_token_usage(total_token_usage, token_usage)

        if tasks is None:
            state["final_answer"] = FinalAnswerEnum.CorrectionError
            state["suggested_fix"] = "N/A"
            return state, total_token_usage

        self._print_step(state, step_output=f"Correction plan: {len(tasks)} task(s)")
        for i, task in enumerate(tasks):
            self._print_step(state, step_output=f"  Task {i+1}: {task}")

        # Step 2: Execute each task sequentially, chaining the results
        current_md = curated_md
        for i, task in enumerate(tasks):
            self._print_step(state, step_output=f"Executing task {i+1}/{len(tasks)}: {task}")
            result_md, token_usage = self._execute_single_task(
                state, source_tables, current_md, task, task_index=i+1, total_tasks=len(tasks),
            )
            total_token_usage = increase_token_usage(total_token_usage, token_usage)

            if result_md is None:
                logger.error(f"Task {i+1} failed; stopping correction chain.")
                self._print_step(state, step_output=f"Task {i+1} failed; stopping correction chain. Applying partial corrections.")
                break
            current_md = result_md

        if current_md == curated_md:
            logger.warning("Correction step produced no changes; passing unchanged table back to verification.")
            self._print_step(state, step_output="Correction step produced no changes; passing unchanged table back to verification.")

        state["curated_table"] = current_md
        self._print_step(state, step_output=f"Corrected Table: \n\n{current_md}")
        return state, total_token_usage

    # ------------------------------------------------------------------
    # Step 1: Planning
    # ------------------------------------------------------------------

    def _plan_corrections(
        self,
        state: PKPECurationWorkflowState,
        source_tables: str,
        reasoning_process: str,
        curated_md: str,
    ) -> tuple[Optional[list[str]], dict]:
        system_prompt = PKPE_CORRECTION_PLANNING_PROMPT.format(
            paper_title=state["paper_title"],
            source_tables=source_tables,
            curated_table=curated_md,
            reasoning_process=reasoning_process,
            domain=self.domain,
        )
        agent = self.get_agent(llm=self.llm)
        try:
            res, _, token_usage, reasoning = agent.go(
                system_prompt=system_prompt,
                instruction_prompt="Decompose the corrections into small tasks.",
                schema=PKPECorrectionPlanResult,
            )
            if reasoning is None:
                reasoning = "N/A"
            self._print_step(state, step_output=f"Planning reasoning: {reasoning}")
            return res.tasks, token_usage
        except Exception as e:
            logger.error(f"Planning step failed: {e}")
            self._print_step(state, step_output=f"Planning step failed: {e}")
            return None, {**DEFAULT_TOKEN_USAGE}

    # ------------------------------------------------------------------
    # Step 2: Per-task code generation and execution
    # ------------------------------------------------------------------

    def _execute_single_task(
        self,
        state: PKPECurationWorkflowState,
        source_tables: str,
        curated_md: str,
        task_description: str,
        task_index: int,
        total_tasks: int,
    ) -> tuple[Optional[str], dict]:
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        error_history = []

        for attempt in range(MAX_CODE_RETRIES):
            system_prompt = PKPE_CORRECTION_TASK_PROMPT.format(
                paper_title=state["paper_title"],
                source_tables=source_tables,
                curated_table=curated_md,
                task_description=task_description,
                domain=self.domain,
            )

            if error_history:
                error_context = "\n\n".join([f"Attempt {i+1} Error: {err}" for i, err in enumerate(error_history)])
                system_prompt += f"\n\n### Previous Execution Errors\n{error_context}\n\nPlease fix the code based on these errors."
                if self._has_truncation_error(error_history):
                    system_prompt += (
                        "\n\n### TRUNCATION WARNING\n"
                        "Your previous code was TRUNCATED because it exceeded the token limit. "
                        "You MUST use a shorter approach. Use df.at or df.loc for targeted edits.\n"
                    )

            instruction_prompt = (
                f"Generate code for task {task_index}/{total_tasks}."
                if attempt == 0
                else f"Retry {attempt + 1}/{MAX_CODE_RETRIES} for task {task_index}/{total_tasks}."
            )

            agent = self.get_agent(llm=self.llm)

            try:
                res, _, token_usage, reasoning_process = agent.go(
                    system_prompt=system_prompt,
                    instruction_prompt=instruction_prompt,
                    schema=PKPECorrectionStepResult,
                )

                if token_usage:
                    total_token_usage = increase_token_usage(total_token_usage, token_usage)

                self._print_step(state, step_output=reasoning_process if reasoning_process is not None else "N / A")
                code = res.code.strip()
                self._print_step(
                    state,
                    step_output=f"Generated Code (Task {task_index}, Attempt {attempt + 1}):\n\n```python\n{code}\n```",
                )

                df_corrected, execution_error = self._execute_code_and_extract_dataframe(code, curated_md)

                if execution_error is not None:
                    error_history.append(execution_error)
                    self._print_step(
                        state,
                        step_output=f"Execution Error (Task {task_index}, Attempt {attempt + 1}):\n\n{execution_error}",
                    )
                    logger.error(f"Task {task_index} code execution failed (attempt {attempt + 1}): {execution_error}")
                    continue

                assert df_corrected is not None, "df_corrected should not be None if execution_error is None"
                if not isinstance(df_corrected, pd.DataFrame):
                    error_msg = f"df_corrected is not a pandas DataFrame, got {type(df_corrected)}"
                    error_history.append(error_msg)
                    logger.error(f"Task {task_index} failed (attempt {attempt + 1}): {error_msg}")
                    continue

                corrected_table_md = dataframe_to_markdown(df_corrected)
                try:
                    markdown_to_dataframe(corrected_table_md)
                except Exception as e:
                    error_msg = f"Generated markdown table is invalid: {e}"
                    error_history.append(error_msg)
                    logger.error(f"Task {task_index} failed (attempt {attempt + 1}): {error_msg}")
                    continue

                self._print_step(
                    state,
                    step_output=f"Task {task_index} succeeded (shape: {df_corrected.shape})",
                )
                return corrected_table_md, total_token_usage

            except RetryException as e:
                logger.error(f"RetryException in task {task_index}: {e}")
                self._print_step(state, step_output=f"RetryException in task {task_index}: {e}")
                break
            except Exception as e:
                error_msg = f"Unexpected error: {type(e).__name__}: {e}"
                error_history.append(error_msg)
                logger.error(f"Task {task_index} failed (attempt {attempt + 1}): {error_msg}")
                continue

        last_error = error_history[-1] if error_history else "unknown error"
        logger.error(f"Task {task_index} failed after {MAX_CODE_RETRIES} attempts. Last error: {last_error}")
        self._print_step(state, step_output=f"Task {task_index} failed after {MAX_CODE_RETRIES} attempts. Last error:\n\n{last_error}")
        return None, total_token_usage

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def _execute_code_and_extract_dataframe(self, code: str, curated_md: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        python_tool = CustomPythonAstREPLTool()
        python_tool.set_runtime(
            curated_md=curated_md,
            markdown_to_dataframe=markdown_to_dataframe,
        )

        execution_output = python_tool._run(code)

        if execution_output.startswith("[ERROR]"):
            return None, execution_output

        df_corrected = getattr(python_tool, "_exec_globals", {}).get("df_corrected", None)

        if df_corrected is None:
            error_msg = f"[ERROR] df_corrected was not created by the code.\n\nCaptured output:\n{execution_output}"
            return None, error_msg

        return df_corrected, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_truncation_error(error_history: list[str]) -> bool:
        truncation_patterns = [
            "was never closed",
            "unexpected EOF while parsing",
            "unterminated string literal",
        ]
        return any(
            pattern in err
            for err in error_history
            for pattern in truncation_patterns
        )

    def leave_step(self, state, token_usage: Optional[dict[str, int]] = None):
        return super().leave_step(state, token_usage)
