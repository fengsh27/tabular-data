import os
from pathlib import Path
import logging

from databases.pmid_db import PMIDDB

logger = logging.getLogger(__name__)

def get_pmid_db():
    db_path = os.environ.get("DATA_FOLDER", "./data")
    db_path = Path(db_path, "databases")
    try:
        os.makedirs(db_path, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create db path: {e}")
        raise e
    db_path = db_path / "pmid_info.db"
    return PMIDDB(db_path)
