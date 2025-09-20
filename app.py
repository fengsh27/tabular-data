import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
import asyncio
import time
from datetime import datetime
import logging
from dotenv import load_dotenv

from databases.constants import CurationJobResponse, CurationRequest, JobResponse, JobStatus
from databases.db_utils import get_pmid_db
from databases.job_db import CurationJob, CurationJobsDB, get_curation_jobs_db
from databases.job_result_db import CurationJobResultDB, get_curation_job_result_db
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents_manager.pk_pe_manager import PKPEManager
from extractor.request_openai import get_5_openai

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="Data Curation API", version="1.0.0")
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# In-memory job storage (in production, use Redis or database)
jobs: Dict[str, JobResponse] = {}

## ────────────────────────────────────────────────────────────────────────────
## Helper functions
## ────────────────────────────────────────────────────────────────────────────

def get_job_db() -> CurationJobsDB:
    db_path = os.environ.get("DATA_FOLDER", "./data")
    db_path = Path(db_path, "databases")
    try:
        os.makedirs(db_path, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create db path: {e}")
        raise e
    db_path = db_path / "job_db.db"
    return get_curation_jobs_db(db_path)

def get_job_result_db() -> CurationJobResultDB:
    db_path = os.environ.get("DATA_FOLDER", "./data")
    db_path = Path(db_path, "databases")
    try:
        os.makedirs(db_path, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create db path: {e}")
        raise e
    db_path = db_path / "job_result_db.db"
    return get_curation_job_result_db(db_path)

async def check_running_jobs(job_db: CurationJobsDB, job: CurationJob):
    while True:
        running_jobs = job_db.get_jobs_by_status(JobStatus.RUNNING.value)
        while len(running_jobs) > 0:
            await asyncio.sleep(1)
            running_jobs = job_db.get_jobs_by_status(JobStatus.RUNNING.value)
    
        pending_jobs = job_db.get_jobs_by_status(JobStatus.PENDING.value)
        if len(pending_jobs) == 0:
            return
    
        if len(pending_jobs) > 0:
            pending_jobs.sort(key=lambda x: x.createAt)
            if job.createAt <= pending_jobs[0].createAt:
                return
            await asyncio.sleep(1)
 

async def curation_work(
    job_id: str, 
    pmid: str,
    mode: str, 
    pipelines: Optional[List[str]] = None
):
    token_usage_acc: dict[str, int] = {**DEFAULT_TOKEN_USAGE}
    def step_callback(
        step_name: str, 
        step_description: str, 
        step_output: str, 
        step_reasoning_process: str, 
        token_usage: dict[str, int]
    ):
        nonlocal token_usage_acc
        if token_usage is not None:
            logger.info(
                f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
            )
            token_usage_acc = increase_token_usage(token_usage_acc, token_usage)
            logger.info(
                f"overall total tokens: {token_usage_acc['total_tokens']}, overall prompt tokens: {token_usage_acc['prompt_tokens']}, overall completion tokens: {token_usage_acc['completion_tokens']}"
            )
        if step_name:
            logger.info("=" * 64)
            logger.info(step_name)
        if step_description:
            logger.info(step_description)
        if step_output:
            logger.info(step_output)
        if step_reasoning_process:
            logger.info(f"\n\n{step_reasoning_process}\n\n")
    def curation_start_callback(pmid: str, job_name: str | None = None):
        pass
    def curation_end_callback(pmid: str, job_name: str, result: PKPECuratedTables):
        pass
    job_db = get_job_db()
    job = job_db.get_job_by_id(job_id)
    if job is None:
        raise Exception(f"Job {job_id} not found")
    if job.status != JobStatus.PENDING.value:
        raise Exception(f"Job {job_id} is not pending")
    await check_running_jobs(job_db, job)

    job_db.update_job_status(job_id, JobStatus.RUNNING.value)
    job_db.set_job_run_time(job_id)
    try:
        pmid_db = get_pmid_db()
        llm = get_5_openai()
        mgr = PKPEManager(llm, pmid_db)
        mgr.run(pmid, curation_start_callback=None, curation_end_callback=None, pipeline_types=pipelines)
        
        # Complete the job
        job_db.complete_job(job_id, JobStatus.COMPLETED.value, "Success")
        job_result_db = get_job_result_db()
        job_result_db.insert_result(job_id, job.session_id, JobStatus.COMPLETED.value, "Success", "Success")
        
    except Exception as e:
        # Handle job failure
        job_db.complete_job(job_id, JobStatus.FAILED.value, str(e))
        job_result_db = get_job_result_db()
        job_result_db.insert_result(job_id, job.session_id, JobStatus.FAILED.value, str(e), str(e))

## ────────────────────────────────────────────────────────────────────────────
## API endpoints
## ────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/curate", response_model=CurationJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_curation(request: CurationRequest, background_tasks: BackgroundTasks):
    """
    Start a data curation job
    
    Args:
        request: Curation request containing mode and optional pipelines
        
    Returns:
        Job ID and confirmation message
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Validate curation mode (add your valid modes here)
    valid_modes = ["auto-mode", "customize-mode"]
    if request.curation_mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid curation mode. Valid modes: {valid_modes}"
        )
    
    # Create job record
    job = JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING.value,
        created_at=datetime.now(),
        message=f"Job created for mode: {request.curation_mode}"
    )
    jobs[job_id] = job

    job_db = get_job_db()
    job_db.insert_job(job_id, request.pmid, request.session_id, JobStatus.PENDING.value)
    
    # Start background task
    background_tasks.add_task(
        curation_work, 
        job_id, 
        request.pmid,
        request.curation_mode, 
        request.curation_pipelines,
    )
    
    return CurationJobResponse(
        job_id=job_id,
        message=f"Curation job started with ID: {job_id}"
    )

@app.get("/api/v1/status/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a curation job
    
    Args:
        job_id: The unique identifier of the job
        
    Returns:
        Job status information including progress and results
    """
    job_db = get_job_db()
    job = job_db.get_job_by_id(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )
    return JobResponse(
        job_id=job_id,
        status=job.status,
        created_at=job.createAt,
        started_at=job.runAt,
        completed_at=job.completeAt,
        progress=0,
        message="",
        error="",
        result={}
    )
    

@app.get("/api/v1/jobs/{session_id}")
async def list_all_jobs(session_id: str):
    """
    List all jobs (useful for debugging/monitoring)
    """
    job_db = get_job_db()
    jobs = job_db.get_jobs_by_session_id(session_id)
    return {"jobs": jobs}

@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job record
    """
    job_db = get_job_db()
    res = job_db.delete_job(job_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {"message": "Data Curation API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint with more details
    """
    job_db = get_job_db()
    active_jobs = job_db.get_jobs_by_status(JobStatus.RUNNING.value)
    total_jobs = job_db.get_all_jobs()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(active_jobs),
        "total_jobs": len(total_jobs)
    }

app.add_websocket_route("/ws/curation", curation_websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
