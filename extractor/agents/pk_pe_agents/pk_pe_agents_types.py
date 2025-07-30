
from pydantic import BaseModel, Field
from typing import Optional, TypedDict

class PKPECurationWorkflowState(TypedDict):
    paper_title: str
    source_tables: str
    curated_table: str
    final_answer: Optional[str] = None
    suggested_fix: Optional[str] = None
    explanation: Optional[str] = None
    intermediate_output: Optional[str] = None



