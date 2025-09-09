import sqlite3
from sqlite3 import Connection
from pathlib import Path
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

curation_job_result_table_name = "CurationJobResult"

class CurationJobResult(BaseModel):
    job_id: str
    session_id: str
    pipeline_complete_status: str
    pipeline_results: str
    pipeline_logs: str

    @classmethod
    def from_db_row(cls, row: dict) -> 'CurationJobResult':
        """Create CurationJobResult from database row."""
        return cls(
            job_id=row['job_id'],
            session_id=row['session_id'],
            pipeline_complete_status=row['pipeline_complete_status'],
            pipeline_results=row['pipeline_results'],
            pipeline_logs=row['pipeline_logs']
        )

curation_job_result_table_schema = f"""
CREATE TABLE IF NOT EXISTS {curation_job_result_table_name} (
    job_id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    pipeline_complete_status TEXT NOT NULL,
    pipeline_results TEXT NOT NULL,
    pipeline_logs TEXT NOT NULL
)
"""

curation_job_result_insert_schema = f"""
INSERT INTO {curation_job_result_table_name} (job_id, session_id, pipeline_complete_status, pipeline_results, pipeline_logs) 
VALUES (?, ?, ?, ?, ?)
"""

curation_job_result_update_schema = f"""
UPDATE {curation_job_result_table_name} 
SET session_id = ?, pipeline_complete_status = ?, pipeline_results = ?, pipeline_logs = ?
WHERE job_id = ?
"""

curation_job_result_select_by_job_id_schema = f"""
SELECT * FROM {curation_job_result_table_name} WHERE job_id = ?
"""

curation_job_result_select_by_session_id_schema = f"""
SELECT * FROM {curation_job_result_table_name} WHERE session_id = ?
"""

curation_job_result_select_all_schema = f"""
SELECT * FROM {curation_job_result_table_name} ORDER BY job_id
"""

curation_job_result_delete_schema = f"""
DELETE FROM {curation_job_result_table_name} WHERE job_id = ?
"""


class CurationJobResultDB:
    """Database manager for CurationJobResult table.
    
    The CurationJobResult table stores the results of curation jobs with the following fields:
    - job_id: Unique identifier for the job (primary key)
    - session_id: Session identifier for grouping jobs
    - pipeline_complete_status: Status of the pipeline completion
    - pipeline_results: Results from the pipeline execution
    - pipeline_logs: Logs from the pipeline execution
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.conn: Optional[Connection] = None
        self.db_path = db_path
        
    def _ensure_table(self) -> bool:
        """Ensure the CurationJobResult table exists."""
        if self.conn is None:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_job_result_table_schema)
            self.conn.commit()
            logger.info("CurationJobResult table ensured successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to ensure CurationJobResult table: {e}")
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
        return db_path / "curation_job_result.db"
        
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

    def insert_result(self, job_id: str, session_id: str, pipeline_complete_status: str, 
                     pipeline_results: str, pipeline_logs: str) -> bool:
        """Insert a new curation job result."""
        if not self._connect_to_db():
            return False
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_job_result_insert_schema, 
                         (job_id, session_id, pipeline_complete_status, pipeline_results, pipeline_logs))
            self.conn.commit()
            logger.info(f"Inserted result for job {job_id} with session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert result for job {job_id}: {e}")
            return False

    def update_result(self, job_id: str, session_id: str, pipeline_complete_status: str, 
                     pipeline_results: str, pipeline_logs: str) -> bool:
        """Update an existing curation job result."""
        if not self._connect_to_db():
            return False
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_job_result_update_schema, 
                         (session_id, pipeline_complete_status, pipeline_results, pipeline_logs, job_id))
            self.conn.commit()
            logger.info(f"Updated result for job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update result for job {job_id}: {e}")
            return False

    def get_result_by_job_id(self, job_id: str) -> Optional[CurationJobResult]:
        """Get a curation job result by job_id."""
        if not self._connect_to_db():
            return None
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_job_result_select_by_job_id_schema, (job_id,))
            row = cursor.fetchone()
            if row:
                return CurationJobResult.from_db_row(dict(row))
            return None
        except Exception as e:
            logger.error(f"Failed to get result for job {job_id}: {e}")
            return None

    def get_results_by_session_id(self, session_id: str) -> List[CurationJobResult]:
        """Get all curation job results for a specific session_id."""
        if not self._connect_to_db():
            return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_job_result_select_by_session_id_schema, (session_id,))
            rows = cursor.fetchall()
            return [CurationJobResult.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get results for session_id {session_id}: {e}")
            return []

    def get_all_results(self) -> List[CurationJobResult]:
        """Get all curation job results ordered by job_id."""
        if not self._connect_to_db():
            return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_job_result_select_all_schema)
            rows = cursor.fetchall()
            return [CurationJobResult.from_db_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get all results: {e}")
            return []

    def delete_result(self, job_id: str) -> bool:
        """Delete a curation job result by job_id."""
        if not self._connect_to_db():
            return False
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_job_result_delete_schema, (job_id,))
            self.conn.commit()
            logger.info(f"Deleted result for job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete result for job {job_id}: {e}")
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


def get_curation_job_result_db(db_path: Optional[Path] = None) -> CurationJobResultDB:
    """Get a CurationJobResultDB instance with optional custom db_path."""
    return CurationJobResultDB(db_path)
