
from pydantic import BaseModel, Field
from typing import Callable, Literal, Optional, TypedDict
from enum import Enum

class PaperTypeEnum(Enum):
    PK = "PK"
    PE = "PE"
    Both = "Both"
    Neither = "Neither"
    Unknown = "Unknown"

class PKPECurationWorkflowState(TypedDict):
    pmid: str
    paper_type: Optional[PaperTypeEnum] = PaperTypeEnum.Unknown
    paper_title: str
    paper_abstract: str
    full_text: Optional[str] = None
    source_tables: Optional[str] = None
    curated_table: Optional[str] = None
    final_answer: Optional[bool] = None
    suggested_fix: Optional[str] = None
    explanation: Optional[str] = None
    verification_reasoning_process: Optional[str] = None
    previous_errors: Optional[str] = None
    step_output_callback: Optional[Callable] = None
    step_count: Optional[int] = 0
    pipeline_tools: Optional[list[str]] = None

class PKPECuratedTables(TypedDict):
    correct: bool
    curated_table: Optional[str] = None
    explanation: Optional[str] = None
    suggested_fix: Optional[str] = None




