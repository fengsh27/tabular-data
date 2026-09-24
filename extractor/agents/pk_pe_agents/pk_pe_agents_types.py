
from pydantic import BaseModel, Field
from typing import Callable, Literal, Optional, TypedDict
from enum import Enum

class PaperTypeEnum(Enum):
    PK = "PK"
    PE = "PE"
    Both = "Both"
    Neither = "Neither"
    Unknown = "Unknown"


class VerifyScopeEnum(Enum):
    """Where the verify -> correct loop looks at the curated data.

    Combined: the tool's per-table results are concatenated first, then verified/corrected
              as one table against all source tables at once. Was the only behaviour before
              PerTable existed; still used automatically as PerTable's fallback (below).
    PerTable: (default) each table is verified/corrected on its own (scoped to just that
              table and its own source), before being combined - cheaper, and mistakes in
              one table can't corrupt another's row indices, but a check that only makes
              sense across tables (duplicate rows between tables, Patient ID numbering
              consistency, a row-scope rule such as "dose rows don't belong in the
              individual-concentration table") is out of view for it. Falls back to Combined
              when the tool doesn't populate state["curated_tables"] (only
              PKIndividualTablesCurationTool does, initially - every other pipeline gets
              Combined regardless of this setting, until it's upgraded too).
    """
    Combined = "combined"
    PerTable = "per_table"


class FinalAnswerEnum(Enum):
    Correct = "Correct"
    Incorrect = "Incorrect"
    MaxStepReached = "MaxStepReached"
    NoTable = "NoTable"
    NoIndividualData = "NoIndividualData"
    PipelineError = "PipelineError"
    CorrectionError = "CorrectionError"
    VerificationError = "VerificationError"
    # A curated table was produced but the verification/correction loop was
    # intentionally skipped (pipeline mode) - not a judgment of correctness.
    Unverified = "Unverified"

    @property
    def is_terminal(self) -> bool:
        """True for any state that ends the verify/correct loop.
        Only Incorrect continues; everything else terminates."""
        return self != FinalAnswerEnum.Incorrect

class PKPECurationWorkflowState(TypedDict):
    pmid: str
    paper_type: Optional[PaperTypeEnum] = PaperTypeEnum.Unknown
    paper_title: str
    paper_abstract: str
    full_text: Optional[str] = None
    # Every tool now returns a list[str] here (a single-input tool returns a 1-element
    # list); format_source_tables() still accepts a bare str too, for the rare early-exit
    # path (pmid_info missing) where a tool returns (None, None).
    source_tables: Optional[list[str] | str] = None
    curated_table: Optional[str] = None
    # Per-table markdown, same order/length as source_tables. Every tool populates this now
    # (see pk_pe_agent_tools.py); None stays possible only as the unsupported-tool signal
    # PKPEExecutionStep falls back to Combined on, which is currently unreachable but kept
    # as a safety net (see tool_supports_per_table in pk_pe_agenttool_task.py).
    curated_tables: Optional[list[str]] = None
    final_answer: Optional[FinalAnswerEnum] = None
    suggested_fix: Optional[str] = None
    explanation: Optional[str] = None
    verification_reasoning_process: Optional[str] = None
    previous_errors: Optional[str] = None
    previous_verification_thoughts: Optional[list[str]] = None
    step_output_callback: Optional[Callable] = None
    step_count: Optional[int] = 0
    pipeline_tools: Optional[list[str]] = None

class PKPECuratedTables(TypedDict):
    correct: FinalAnswerEnum
    curated_table: Optional[str]
    explanation: Optional[str]
    suggested_fix: Optional[str]




