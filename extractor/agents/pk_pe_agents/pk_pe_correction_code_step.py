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

class PKPECuratedTablesCorrectionCodeStep(PKPECommonStep):
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
            instruction_prompt="Let's start generating the code to correct the curated table.",# instruction_prompt,
            schema=PKPECorrectionStepResult,
        )
        self._print_step(state, step_output=reasoning_process if reasoning_process is not None else "N / A")
        res: PKPECorrectionStepResult = res
        self._print_step(state, step_output=f"Corrected Table: \n\n{res.corrected_table}")
        state["curated_table"] = res.corrected_table

        return state, token_usage

    def leave_step(self, state, token_usage: Optional[dict[str, int]] = None):
        return super().leave_step(state, token_usage)

