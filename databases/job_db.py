import sqlite3
from sqlite3 import Connection
from pathlib import Path
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, field_validator
from datetime import datetime

logger = logging.getLogger(__name__)

curation_jobs_table_name = "CurationJobs"

class CurationJob(BaseModel):
    job_id: str
    pmid: str
    session_id: str
    createAt: datetime
    status: str
    runAt: datetime | None = None
    completeAt: datetime | None = None
    completeStatus: str | None = None

    @classmethod
    def from_db_row(cls, row: dict) -> 'CurationJob':
        """Create CurationJob from database row with proper type handling."""
        def parse_datetime(dt_str: str | None) -> datetime | None:
            """Parse SQLite datetime string to datetime object."""
            if dt_str is None:
                return None
            # Handle SQLite datetime format: 'YYYY-MM-DD HH:MM:SS.fff'
            return datetime.fromisoformat(dt_str.replace(' ', 'T'))
        
        return cls(
            job_id=row['job_id'],
            pmid=row['pmid'],
            session_id=row['session_id'],
            createAt=parse_datetime(row['createAt']),
            status=row['status'],
            runAt=parse_datetime(row['runAt']),
            completeAt=parse_datetime(row['completeAt']),
            completeStatus=row['completeStatus']
        )

curation_jobs_table_schema = f"""
CREATE TABLE IF NOT EXISTS {curation_jobs_table_name} (
    job_id VARCHAR(64) PRIMARY KEY UNIQUE,
    pmid VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    createAt TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    runAt TEXT DEFAULT NULL,
    completeAt TEXT DEFAULT NULL,
    completeStatus TEXT DEFAULT NULL,
    status VARCHAR(64) NOT NULL DEFAULT 'pending'
)
"""

curation_jobs_insert_schema = f"""
INSERT INTO {curation_jobs_table_name} (job_id, pmid, session_id, status) 
VALUES (?, ?, ?, ?)
"""

curation_jobs_update_schema = f"""
UPDATE {curation_jobs_table_name} 
SET status = ?, completeAt = strftime('%Y-%m-%d %H:%M:%f', 'now')
WHERE job_id = ?
"""

curation_jobs_set_run_time_schema = f"""
UPDATE {curation_jobs_table_name} 
SET runAt = strftime('%Y-%m-%d %H:%M:%f', 'now')
WHERE job_id = ?
"""

curation_jobs_complete_job_schema = f"""
UPDATE {curation_jobs_table_name} 
SET status = ?, completeAt = strftime('%Y-%m-%d %H:%M:%f', 'now'), completeStatus = ?
WHERE job_id = ?
"""

curation_jobs_update_status_only_schema = f"""
UPDATE {curation_jobs_table_name} 
SET status = ?
WHERE job_id = ?
"""

curation_jobs_select_by_job_id_schema = f"""
SELECT * FROM {curation_jobs_table_name} WHERE job_id = ?
"""

curation_jobs_select_by_pmid_schema = f"""
SELECT * FROM {curation_jobs_table_name} WHERE pmid = ?
"""

curation_jobs_select_by_session_id_schema = f"""
SELECT * FROM {curation_jobs_table_name} WHERE session_id = ?
"""

curation_jobs_select_by_status_schema = f"""
SELECT * FROM {curation_jobs_table_name} WHERE status = ?
"""

curation_jobs_select_all_schema = f"""
SELECT * FROM {curation_jobs_table_name} ORDER BY createAt DESC
"""

curation_jobs_delete_schema = f"""
DELETE FROM {curation_jobs_table_name} WHERE job_id = ?
"""


class CurationJobsDB:
    """Database manager for CurationJobs table.
    
    The CurationJobs table tracks curation jobs with the following fields:
    - job_id: Unique identifier for the job
    - pmid: PubMed ID being processed
    - session_id: Session identifier for grouping jobs
    - createAt: When the job was created
    - runAt: When the job started running (NULL if not started)
    - completeAt: When the job completed (NULL if not completed)
    - completeStatus: Completion status like "Failed due to timeout" (NULL if not completed)
    - status: Current job status (pending, running, completed, etc.)
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.conn: Optional[Connection] = None
        self.db_path = db_path
        
    def _ensure_table(self) -> bool:
        """Ensure the CurationJobs table exists."""
        if self.conn is None:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_table_schema)
            self.conn.commit()
            logger.info("CurationJobs table ensured successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to ensure CurationJobs table: {e}")
            return False

    def _get_db_path(self) -> Path:
        """Get the database path, creating directories if needed."""
        if self.db_path is not None:
            return self.db_path
            
        db_path = os.environ.get("DATA_FOLDER", "./data")
        db_path = Path(db_path, "databases")
        try:
            os.makedirs(db_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create db path: {e}")
            raise e
        return db_path / "curation_jobs.db"
        
    def _connect_to_db(self) -> bool:
        """Connect to the SQLite database."""
        if self.conn is not None:
            return True
        
        try:
            db_path = self._get_db_path()
            if not db_path.exists():
                logger.info(f"Creating db file: {db_path}")
                db_path.touch()
            else:
                logger.info(f"Using existing db file: {db_path}")
                
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            self._ensure_table()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to db: {e}")
            return False

    def insert_job(self, job_id: str, pmid: str, session_id: str, status: str = "pending") -> bool:
        """Insert a new curation job."""
        if not self._connect_to_db():
            return False
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_insert_schema, (job_id, pmid, session_id, status))
            self.conn.commit()
            logger.info(f"Inserted job {job_id} for pmid {pmid} with session {session_id} and status {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert job {job_id}: {e}")
            return False

    def update_job_status(self, job_id: str, status: str) -> bool:
        """Update the status of a curation job (without setting completion time)."""
        if not self._connect_to_db():
            return False
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_update_status_only_schema, (status, job_id))
            self.conn.commit()
            logger.info(f"Updated job {job_id} status to {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job_id} status: {e}")
            return False

    def set_job_run_time(self, job_id: str) -> bool:
        """Set the runAt timestamp for a job when it starts running."""
        if not self._connect_to_db():
            return False
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_set_run_time_schema, (job_id,))
            self.conn.commit()
            logger.info(f"Set run time for job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set run time for job {job_id}: {e}")
            return False

    def complete_job(self, job_id: str, status: str, complete_status: Optional[str] = None) -> bool:
        """Mark a job as completed with optional completion status."""
        if not self._connect_to_db():
            return False
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_complete_job_schema, (status, complete_status, job_id))
            self.conn.commit()
            logger.info(f"Completed job {job_id} with status {status} and completion status {complete_status}")
            return True
        except Exception as e:
            logger.error(f"Failed to complete job {job_id}: {e}")
            return False

    def get_job_by_id(self, job_id: str) -> Optional[CurationJob]:
        """Get a curation job by job_id."""
        if not self._connect_to_db():
            return None
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_select_by_job_id_schema, (job_id,))
            row = cursor.fetchone()
            if row:
                return CurationJob.from_db_row(dict(row))
            return None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None

    def get_jobs_by_pmid(self, pmid: str) -> List[CurationJob]:
        """Get all curation jobs for a specific pmid."""
        if not self._connect_to_db():
            return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_select_by_pmid_schema, (pmid,))
            rows = cursor.fetchall()
            return [CurationJob.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get jobs for pmid {pmid}: {e}")
            return []

    def get_jobs_by_session_id(self, session_id: str) -> List[CurationJob]:
        """Get all curation jobs for a specific session_id."""
        if not self._connect_to_db():
            return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_select_by_session_id_schema, (session_id,))
            rows = cursor.fetchall()
            return [CurationJob.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get jobs for session_id {session_id}: {e}")
            return []

    def get_jobs_by_status(self, status: str) -> List[CurationJob]:
        """Get all curation jobs with a specific status."""
        if not self._connect_to_db():
            return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_select_by_status_schema, (status,))
            rows = cursor.fetchall()
            return [CurationJob.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get jobs with status {status}: {e}")
            return []

    def get_all_jobs(self) -> List[CurationJob]:
        """Get all curation jobs ordered by creation time."""
        if not self._connect_to_db():
            return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_select_all_schema)
            rows = cursor.fetchall()
            return [CurationJob.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get all jobs: {e}")
            return []

    def get_jobs_by_complete_status(self, complete_status: str) -> List[CurationJob]:
        """Get all curation jobs with a specific complete status."""
        if not self._connect_to_db():
            return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT * FROM {curation_jobs_table_name} WHERE completeStatus = ?", (complete_status,))
            rows = cursor.fetchall()
            return [CurationJob.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get jobs with complete status {complete_status}: {e}")
            return []

    def get_running_jobs(self) -> List[CurationJob]:
        """Get all jobs that are currently running (have runAt but no completeAt)."""
        if not self._connect_to_db():
            return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT * FROM {curation_jobs_table_name} WHERE runAt IS NOT NULL AND completeAt IS NULL")
            rows = cursor.fetchall()
            return [CurationJob.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get running jobs: {e}")
            return []

    def get_pending_jobs(self) -> List[CurationJob]:
        """Get all jobs that are pending (no runAt timestamp)."""
        if not self._connect_to_db():
            return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT * FROM {curation_jobs_table_name} WHERE runAt IS NULL")
            rows = cursor.fetchall()
            return [CurationJob.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get pending jobs: {e}")
            return []

    def delete_job(self, job_id: str) -> bool:
        """Delete a curation job by job_id."""
        if not self._connect_to_db():
            return False
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_jobs_delete_schema, (job_id,))
            self.conn.commit()
            logger.info(f"Deleted job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    def close_connection(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        self._connect_to_db()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()


def get_curation_jobs_db(db_path: Optional[Path] = None) -> CurationJobsDB:
    """Get a CurationJobsDB instance with optional custom db_path."""
    return CurationJobsDB(db_path)
