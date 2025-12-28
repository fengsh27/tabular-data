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
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState
from extractor.agents.pk_pe_agents.pk_pe_agents_utils import format_source_tables
from extractor.agents.custom_python_ast_repl_tool import CustomPythonAstREPLTool
from extractor.constants import COT_USER_INSTRUCTION

logger = logging.getLogger(__name__)

PKPE_CORRECTION_SYSTEM_PROMPT = """
You are a biomedical data correction engineer with expertise in {domain} and robust Python data wrangling.

You are given:
1) **Paper Title**
2) **Paper Abstract**
3) **Source Table(s) or full text**
4) **Curated Table** (a markdown table string; this is the input to your code)
5) **Reasoning Process** (explains what is wrong and what to fix)
6) **a python function markdown_to_dataframe(md_table: str)** convert markdown table to dataframe

Your task:
- Write **Python code only** that corrects the curated table according to the Reasoning Process and the source table.
- The code must:
  - call provided function markdown_to_dataframe() to convert the curated markdown table string **curated_md** into a pandas DataFrame,
  - apply ONLY the corrections explicitly described in the Reasoning Process (treat it as a patch),
  - preserve all other rows/columns unchanged,
  - output a corrected pandas DataFrame named `df_corrected`.

Hard requirements:
- Output must be a **single Python code block** and nothing else.
- Do NOT include any explanatory text.
- Do NOT make network calls.
- Do NOT read/write files.
- Use only standard libraries plus `pandas` (assume pandas is installed).
- The code must be runnable as-is.

Input/Output contract for your code:
- The curated markdown table will be provided in a variable named `curated_md` (type: str).
- Your code must produce `df_corrected` (type: pandas.DataFrame).
- The DataFrame columns must match the curated table header exactly (same names and order).
- Cell values must remain strings unless the Reasoning Process explicitly requires type conversion.

Parsing requirements:
- The curated markdown table is GitHub-flavored with a header row, a separator row, and body rows.
- Cells are separated by `|`.
- Trim whitespace around cell values.
- Preserve special tokens like "< LOD", "–", "N/A" exactly.

Correction requirements:
- Treat the curated table as the base.
- Treat the Reasoning Process / Suggested Fix as the only source of changes.
- Do not perform “cleanup” such as deduplication unless explicitly required.
- If a correction refers to a row that appears multiple times (duplicate rows), disambiguate by matching as many columns as needed; if still ambiguous, use row position (0-based index in the body) with a clear comment in code.

After applying corrections:
- Validate that:
  - `df_corrected` has the same number of rows as the input unless the Reasoning Process explicitly requires insert/delete,
  - all required corrected values match the Reasoning Process exactly,
  - no unintended rows were changed (e.g., use masks to target only the intended rows).

---

## Output Format
You output **must be** in json compact format, and **exactly match the following schema**:
{{
    "code": <string, the python code that corrects the curated table without fences (like '```python' or '```')>
}}

Now generate the code.

--------------------
### Paper Title
{paper_title}

### Paper Abstract
{paper_abstract}

### Source Table(s) or full text
{source_tables}

### Curated Table
(curated markdown table string will be assigned to `curated_md` in the runtime)

{curated_table}

### Reasoning Process
{reasoning_process}
--------------------


"""

class PKPECorrectionStepResult(BaseModel):
    code: str = Field(description="Python code that corrects the curated table. The code must produce a pandas DataFrame named `df_corrected`.")

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

    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        state: PKPECurationWorkflowState = state
        source_tables = state["source_tables"] if "source_tables" in state else None
        source_tables = format_source_tables(source_tables)
        verification_reasoning_process = state["verification_reasoning_process"] if "verification_reasoning_process" in state else "N / A"
        curated_md = state["curated_table"]
        
        # Retry loop for code generation and execution
        max_retries = 5
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        error_history = []
        
        for attempt in range(max_retries):
            system_prompt = PKPE_CORRECTION_SYSTEM_PROMPT.format(
                paper_title=state["paper_title"],
                paper_abstract=state["paper_abstract"],
                source_tables=source_tables,
                curated_table=curated_md,
                reasoning_process=verification_reasoning_process,
                domain=self.domain,
            )
            
            # Add error history to the prompt if there were previous errors
            if error_history:
                error_context = "\n\n".join([f"Attempt {i+1} Error: {err}" for i, err in enumerate(error_history)])
                system_prompt += f"\n\n### Previous Execution Errors\n{error_context}\n\nPlease fix the code based on these errors."
            
            instruction_prompt = "Let's start generating the code to correct the curated table."
            if attempt > 0:
                instruction_prompt = f"Previous code execution failed. Please regenerate the code fixing the errors. Attempt {attempt + 1}/{max_retries}."

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
                res: PKPECorrectionStepResult = res
                
                # Extract and execute the code
                code = res.code.strip()
                self._print_step(state, step_output=f"Generated Code (Attempt {attempt + 1}):\n\n```python\n{code}\n```")
                
                # Execute the code and extract df_corrected
                df_corrected, execution_error = self._execute_code_and_extract_dataframe(code, curated_md)
                
                # Check if execution was successful
                if execution_error is not None:
                    error_msg = execution_error
                    logger.error(f"Code execution failed (attempt {attempt + 1}): {error_msg}")
                    error_history.append(error_msg)
                    self._print_step(state, step_output=f"Execution Error (Attempt {attempt + 1}):\n\n{error_msg}")
                    
                    if attempt < max_retries - 1:
                        continue  # Retry with error message
                    else:
                        raise RetryException(f"Failed to execute code after {max_retries} attempts. Last error: {error_msg}")
                
                # Validate df_corrected (should not be None if execution_error is None)
                assert df_corrected is not None, "df_corrected should not be None if execution_error is None"
                if not isinstance(df_corrected, pd.DataFrame):
                    error_msg = f"df_corrected is not a pandas DataFrame, got {type(df_corrected)}"
                    logger.error(f"Code execution failed (attempt {attempt + 1}): {error_msg}")
                    error_history.append(error_msg)
                    self._print_step(state, step_output=f"Execution Error (Attempt {attempt + 1}):\n\n{error_msg}")
                    
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        raise RetryException(f"df_corrected is not a valid DataFrame after {max_retries} attempts.")
                
                # Convert df_corrected to markdown
                corrected_table_md = dataframe_to_markdown(df_corrected)
                
                # Validate the markdown table
                try:
                    # Verify it can be parsed back
                    df_verify = markdown_to_dataframe(corrected_table_md)
                    self._print_step(state, step_output=f"Successfully executed code and generated corrected table (shape: {df_corrected.shape})")
                    self._print_step(state, step_output=f"Corrected Table: \n\n{corrected_table_md}")
                    
                    state["curated_table"] = corrected_table_md
                    return state, total_token_usage
                    
                except Exception as e:
                    error_msg = f"Generated markdown table is invalid: {e}"
                    logger.error(f"Code execution failed (attempt {attempt + 1}): {error_msg}")
                    error_history.append(error_msg)
                    self._print_step(state, step_output=f"Execution Error (Attempt {attempt + 1}):\n\n{error_msg}")
                    
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        raise RetryException(f"Failed to generate valid markdown table after {max_retries} attempts.")
                        
            except RetryException:
                # Re-raise RetryException to trigger retry mechanism
                raise
            except Exception as e:
                error_msg = f"Unexpected error: {type(e).__name__}: {e}"
                logger.error(f"Code generation/execution failed (attempt {attempt + 1}): {error_msg}")
                error_history.append(error_msg)
                self._print_step(state, step_output=f"Error (Attempt {attempt + 1}):\n\n{error_msg}")
                
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    raise RetryException(f"Failed after {max_retries} attempts. Last error: {error_msg}")
        
        # Should not reach here, but just in case
        raise RetryException(f"Failed to correct table after {max_retries} attempts.")
    
    def _execute_code_and_extract_dataframe(self, code: str, curated_md: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute the generated code using CustomPythonAstREPLTool and extract df_corrected.

        Returns:
            tuple: (df_corrected, error_message)
            - df_corrected: The pandas DataFrame if execution succeeded, None otherwise
            - error_message: Error message if execution failed, None otherwise
        """
        # Initialize a fresh REPL tool per attempt so that globals don't leak between runs
        python_tool = CustomPythonAstREPLTool()
        python_tool.set_runtime(
            curated_md=curated_md,
            markdown_to_dataframe=markdown_to_dataframe,
        )

        # Run the code; this captures stdout/stderr and returns a textual report
        execution_output = python_tool._run(code)

        # If the tool reports an error, propagate it back
        if execution_output.startswith("[ERROR]"):
            return None, execution_output

        # Pull df_corrected out of the tool's globals
        df_corrected = getattr(python_tool, "_exec_globals", {}).get("df_corrected", None)

        if df_corrected is None:
            # The tool will usually emit a [WARN] line in this case; include it in the message
            error_msg = f"[ERROR] df_corrected was not created by the code.\n\nCaptured output:\n{execution_output}"
            return None, error_msg

        return df_corrected, None

    def leave_step(self, state, token_usage: Optional[dict[str, int]] = None):
        return super().leave_step(state, token_usage)

