import json
from pathlib import Path
import sqlite3
from sqlite3 import Connection
from time import strftime
import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

curation_data_table_name = "CurationData"

curation_data_table_schema = f"""
CREATE TABLE IF NOT EXISTS {curation_data_table_name} (
    pmid VARCHAR(64),
    source_tables VARCHAR(256),
    logs TEXT,
    md_table TEXT,
    curation_type VARCHAR(64),
    token_usage VARCHAR(256),
    modified_time TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    PRIMARY KEY (pmid, curation_type)
)
"""

curation_data_table_insert_schema = f"""
INSERT OR REPLACE INTO {curation_data_table_name} 
(pmid, source_tables, logs, md_table, curation_type, token_usage, modified_time) 
VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%S', 'now'))
"""

curation_data_table_select_schema = f"""
SELECT * FROM {curation_data_table_name} WHERE pmid = ? AND curation_type = ?
"""

curation_data_table_select_all_schema = f"""
SELECT * FROM {curation_data_table_name} ORDER BY modified_time DESC
"""

curation_data_table_delete_schema = f"""
DELETE FROM {curation_data_table_name} WHERE pmid = ? AND curation_type = ?
"""

curation_data_table_update_schema = f"""
UPDATE {curation_data_table_name} 
SET source_tables = ?, logs = ?, md_table = ?, token_usage = ?, 
    modified_time = strftime('%Y-%m-%d %H:%M:%S', 'now')
WHERE pmid = ? AND curation_type = ?
"""

class CurationDB:
    def __init__(self, db_path: Path | None = None):
        self.conn: Connection | None = None
        self.db_path = db_path
        
    def _ensure_table(self):
        if self.conn is None:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_data_table_schema)
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to ensure table: {e}")
            return False

    def _get_db_path(self):
        if self.db_path is not None:
            return self.db_path
        db_path = os.environ.get("DATA_FOLDER", "./data")
        db_path = Path(db_path, "databases")
        try:
            os.makedirs(db_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create db path: {e}")
            raise e
        return db_path / "curation_data.db"
        
    def _connect_to_db(self):
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
            self._ensure_table()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to db: {e}")
            return False

    def insert_curation_data(
        self, 
        pmid: str, 
        source_tables: str, 
        logs: str, 
        md_table: str, 
        curation_type: str, 
        token_usage: str
    ) -> bool:
        """
        Insert or update curation data for a given PMID.
        
        Args:
            pmid: PubMed ID (VARCHAR(64))
            source_tables: Source table information (VARCHAR(256))
            logs: Log information (TEXT)
            md_table: Markdown table content (TEXT)
            curation_type: Type of curation (VARCHAR(64))
            token_usage: Token usage information (VARCHAR(256))
            
        Returns:
            bool: True if successful, False otherwise
        """
        res = self._connect_to_db()
        if not res:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                curation_data_table_insert_schema, 
                (pmid, source_tables, logs, md_table, curation_type, token_usage)
            )
            self.conn.commit()
            logger.info(f"Successfully inserted curation data for PMID: {pmid}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert curation data: {e}")
            return False
        finally:
            self.conn.close()
            self.conn = None

    def update_curation_data(
        self, 
        pmid: str, 
        curation_type: str,
        source_tables: str, 
        logs: str, 
        md_table: str, 
        token_usage: str
    ) -> bool:
        """
        Update existing curation data for a given PMID and curation type.
        
        Args:
            pmid: PubMed ID (VARCHAR(64))
            curation_type: Type of curation (VARCHAR(64))
            source_tables: Source table information (VARCHAR(256))
            logs: Log information (TEXT)
            md_table: Markdown table content (TEXT)
            token_usage: Token usage information (VARCHAR(256))
            
        Returns:
            bool: True if successful, False otherwise
        """
        res = self._connect_to_db()
        if not res:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                curation_data_table_update_schema, 
                (source_tables, logs, md_table, token_usage, pmid, curation_type)
            )
            self.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Successfully updated curation data for PMID: {pmid}, type: {curation_type}")
                return True
            else:
                logger.warning(f"No curation data found for PMID: {pmid}, type: {curation_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to update curation data: {e}")
            return False
        finally:
            self.conn.close()
            self.conn = None

    def select_curation_data(self, pmid: str, curation_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve curation data for a given PMID and curation type.
        
        Args:
            pmid: PubMed ID
            curation_type: Type of curation
            
        Returns:
            Dict containing curation data or None if not found
        """
        res = self._connect_to_db()
        if not res:
            return None
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_data_table_select_schema, (pmid, curation_type))
            row = cursor.fetchone()
            if row is None:
                return None
            
            return {
                "pmid": row[0],
                "source_tables": row[1],
                "logs": row[2],
                "md_table": row[3],
                "curation_type": row[4],
                "token_usage": row[5],
                "modified_time": row[6]
            }
        except Exception as e:
            logger.error(f"Failed to select curation data: {e}")
            return None
        finally:
            self.conn.close()
            self.conn = None

    def select_all_curation_data(self) -> List[Dict[str, Any]]:
        """
        Retrieve all curation data ordered by modified_time (most recent first).
        
        Returns:
            List of dictionaries containing curation data
        """
        res = self._connect_to_db()
        if not res:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_data_table_select_all_schema)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                result.append({
                    "pmid": row[0],
                    "source_tables": row[1],
                    "logs": row[2],
                    "md_table": row[3],
                    "curation_type": row[4],
                    "token_usage": row[5],
                    "modified_time": row[6]
                })
            
            return result
        except Exception as e:
            logger.error(f"Failed to select all curation data: {e}")
            return []
        finally:
            self.conn.close()
            self.conn = None

    def delete_curation_data(self, pmid: str, curation_type: str) -> bool:
        """
        Delete curation data for a given PMID and curation type.
        
        Args:
            pmid: PubMed ID
            curation_type: Type of curation
            
        Returns:
            bool: True if successful, False otherwise
        """
        res = self._connect_to_db()
        if not res:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(curation_data_table_delete_schema, (pmid, curation_type))
            self.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Successfully deleted curation data for PMID: {pmid}, type: {curation_type}")
                return True
            else:
                logger.warning(f"No curation data found for PMID: {pmid}, type: {curation_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete curation data: {e}")
            return False
        finally:
            self.conn.close()
            self.conn = None

    def get_curation_count(self) -> int:
        """
        Get the total number of curation records in the database.
        
        Returns:
            int: Number of records
        """
        res = self._connect_to_db()
        if not res:
            return 0
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {curation_data_table_name}")
            count = cursor.fetchone()[0]
            return count
        except Exception as e:
            logger.error(f"Failed to get curation count: {e}")
            return 0
        finally:
            self.conn.close()
            self.conn = None

    def select_curation_data_by_pmid(self, pmid: str) -> List[Dict[str, Any]]:
        """
        Retrieve all curation data for a given PMID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            List of dictionaries containing curation data for the PMID
        """
        res = self._connect_to_db()
        if not res:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                f"SELECT * FROM {curation_data_table_name} WHERE pmid = ? ORDER BY modified_time DESC",
                (pmid,)
            )
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                result.append({
                    "pmid": row[0],
                    "source_tables": row[1],
                    "logs": row[2],
                    "md_table": row[3],
                    "curation_type": row[4],
                    "token_usage": row[5],
                    "modified_time": row[6]
                })
            
            return result
        except Exception as e:
            logger.error(f"Failed to select curation data by PMID: {e}")
            return []
        finally:
            self.conn.close()
            self.conn = None

    def search_by_curation_type(self, curation_type: str) -> List[Dict[str, Any]]:
        """
        Search curation data by curation type.
        
        Args:
            curation_type: Type of curation to search for
            
        Returns:
            List of dictionaries containing matching curation data
        """
        res = self._connect_to_db()
        if not res:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                f"SELECT * FROM {curation_data_table_name} WHERE curation_type = ? ORDER BY modified_time DESC",
                (curation_type,)
            )
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                result.append({
                    "pmid": row[0],
                    "source_tables": row[1],
                    "logs": row[2],
                    "md_table": row[3],
                    "curation_type": row[4],
                    "token_usage": row[5],
                    "modified_time": row[6]
                })
            
            return result
        except Exception as e:
            logger.error(f"Failed to search by curation type: {e}")
            return []
        finally:
            self.conn.close()
            self.conn = None
