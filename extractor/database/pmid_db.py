
import json
from pathlib import Path
import sqlite3
from sqlite3 import Connection
from time import strftime
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

pmid_info_table_name = "pmid_info"

pmid_info_table_schema = f"""
CREATE TABLE IF NOT EXISTS {pmid_info_table_name} (
    pmid TEXT PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    full_text TEXT,
    tables_json TEXT,
    sections_json TEXT,
    datetime TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
)
"""

pmid_info_table_insert_schema = f"""
INSERT INTO {pmid_info_table_name} (pmid, title, abstract, full_text, tables_json, sections_json, datetime) 
VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%S', 'now'))
"""

pmid_info_table_select_schema = f"""
SELECT * FROM {pmid_info_table_name} WHERE pmid = ?
"""

class PMIDDB:
    def __init__(self, db_path: Path | None = None):
        self.conn: Connection | None = None
        self.db_path = db_path
        
    def _ensure_table(self):
        if self.conn is None:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(pmid_info_table_schema)
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
        return db_path / "pmid_info.db"
        
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

    def insert_pmid_info(
        self, 
        pmid: str, 
        title: str, 
        abstract: str, 
        full_text: str, 
        tables: list[dict], 
        sections: list[str],
    ):
        for table in tables:
            if table["table"] is not None:
                table["table"] = table["table"].to_json()
        tables_json = json.dumps(tables)
        sections_json = json.dumps(sections)
        res = self._connect_to_db()
        if not res:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(pmid_info_table_insert_schema, (pmid, title, abstract, full_text, tables_json, sections_json))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to insert pmid info: {e}")
            return False
        finally:
            self.conn.close()
            self.conn = None
        
    def select_pmid_info(self, pmid: str) -> tuple[str, str, str, str, list[dict], list[str]] | None:
        """
        return (
            pmid,
            title,
            abstract,
            full_text,
            tables,
            sections
        )
        """
        res = self._connect_to_db()
        if not res:
            return None
        try:
            cursor = self.conn.cursor()
            cursor.execute(pmid_info_table_select_schema, (pmid,))
            row = cursor.fetchone()
            if row is None:
                return None
            tables = json.loads(row[4])
            for table in tables:
                table["table"] = pd.read_json(table["table"])
            sections = json.loads(row[5])
            return row[0], row[1], row[2], row[3], tables, sections
        except Exception as e:
            logger.error(f"Failed to select pmid info: {e}")
            return None
        finally:
            self.conn.close()
            self.conn = None
