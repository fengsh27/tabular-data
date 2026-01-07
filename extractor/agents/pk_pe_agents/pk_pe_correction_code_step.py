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
1) Paper Title
2) Paper Abstract
3) Source Table(s) or full text
4) Curated Table (a markdown table string; this is the input to your code)
5) Reasoning Process (explains what is wrong and what to fix)
6) A PROVIDED Python function: markdown_to_dataframe(md_table: str) -> pandas.DataFrame

CRITICAL: markdown_to_dataframe() is already implemented and available at runtime.
- You MUST NOT define it, re-implement it, or include any parsing logic that duplicates it.
- Assume it exists in the global scope and call it directly as: df = markdown_to_dataframe(curated_md).

FORBIDDEN (must not appear anywhere in your output code):
- Any function definition or assignment for markdown_to_dataframe, including:
  - "def markdown_to_dataframe"
  - "markdown_to_dataframe ="
  - any custom markdown parsing implementation intended to replace it

If you violate the FORBIDDEN rule, your output is considered invalid.

Your task:
Write Python code ONLY that corrects the curated table according to the Reasoning Process and the source table(s).

Hard requirements:
- Your output must be valid JSON (compact) matching EXACTLY this schema:
  {{"code": "<python code as a single string WITHOUT code fences>"}}
- The code inside "code" must be runnable as-is (no placeholders).
- Do NOT include any explanatory text outside the JSON object.
- Do NOT make network calls.
- Do NOT read/write files.
- Use only standard libraries plus pandas (assume pandas is installed).

Input/Output contract:
- curated_md (str) will be provided at runtime.
- You must produce df_corrected (pandas.DataFrame).
- df_corrected must preserve:
  - the same columns (names and order) as the curated table header
  - the same number of rows as input, unless the Reasoning Process explicitly requires insertion/deletion
- All cell values must remain strings unless the Reasoning Process explicitly requires type conversion.

Patch-only correction rule (highest priority):
- Treat the curated table as the base.
- Apply ONLY the explicit corrections described in the Reasoning Process (no additional cleanup or normalization).
- Preserve all other rows/columns unchanged.
- If a correction targets a duplicate row, disambiguate by matching as many columns as needed; if still ambiguous, use row position (0-based index in the body) and add a brief code comment explaining the choice.

Required structure of the code (enforced order):
1) import pandas as pd
2) df = markdown_to_dataframe(curated_md)
3) Apply the minimal set of edits specified by the Reasoning Process
4) df_corrected = df (or a modified copy), ensuring column order unchanged
5) Lightweight validation assertions:
   - row count unchanged unless explicitly required
   - edits applied only to intended rows (use boolean masks; assert mask.sum() matches expectation)

DO NOT:
- Define markdown_to_dataframe (see FORBIDDEN)
- Re-parse markdown manually
- Change column names
- Reorder rows unless explicitly required
- Convert types unless explicitly required

Now produce the JSON object with the "code" field only.

--------------------
Paper Title:
{paper_title}

Paper Abstract:
{paper_abstract}

Source Table(s) or full text:
{source_tables}

Curated Table:
(curated markdown table string will be assigned to curated_md at runtime)
{curated_table}

Reasoning Process:
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
                        logger.error(f"Max retries reached; leaving curated_table unchanged. Last error: {error_msg}")
                        self._print_step(state, step_output=f"Max retries reached; leaving curated_table unchanged. Last error:\n\n{error_msg}")
                        return state, total_token_usage
                
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
                        logger.error(f"Max retries reached; leaving curated_table unchanged. Last error: {error_msg}")
                        self._print_step(state, step_output=f"Max retries reached; leaving curated_table unchanged. Last error:\n\n{error_msg}")
                        return state, total_token_usage
                
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
                        logger.error(f"Max retries reached; leaving curated_table unchanged. Last error: {error_msg}")
                        self._print_step(state, step_output=f"Max retries reached; leaving curated_table unchanged. Last error:\n\n{error_msg}")
                        return state, total_token_usage
                        
            except RetryException as e:
                logger.error(f"RetryException encountered; leaving curated_table unchanged. Error: {e}")
                self._print_step(state, step_output=f"RetryException encountered; leaving curated_table unchanged.\n\n{e}")
                return state, total_token_usage
            except Exception as e:
                error_msg = f"Unexpected error: {type(e).__name__}: {e}"
                logger.error(f"Code generation/execution failed (attempt {attempt + 1}): {error_msg}")
                error_history.append(error_msg)
                self._print_step(state, step_output=f"Error (Attempt {attempt + 1}):\n\n{error_msg}")
                
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    logger.error(f"Max retries reached; leaving curated_table unchanged. Last error: {error_msg}")
                    self._print_step(state, step_output=f"Max retries reached; leaving curated_table unchanged. Last error:\n\n{error_msg}")
                    return state, total_token_usage
        
        # Should not reach here, but just in case
        logger.error("Max retries reached; leaving curated_table unchanged.")
        self._print_step(state, step_output="Max retries reached; leaving curated_table unchanged.")
        return state, total_token_usage
    
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
