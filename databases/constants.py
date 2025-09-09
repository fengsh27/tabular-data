from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Pydantic models for request/response
class CurationRequest(BaseModel):
    curation_mode: str
    session_id: str
    pmid: str
    curation_pipelines: Optional[List[str]] = None

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[int] = None  # percentage 0-100
    message: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class CurationJobResponse(BaseModel):
    job_id: str
    message: str
