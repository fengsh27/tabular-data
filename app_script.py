import argparse
import csv
import logging
import os
from pathlib import Path
import time
from dotenv import load_dotenv

from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECuratedTables
from extractor.log_utils import initialize_logger
from extractor.agents_manager.pk_pe_manager import PKPEManager
from extractor.database.pmid_db import PMIDDB
from extractor.request_deepseek import get_deepseek
from extractor.request_geminiai import get_gemini
from extractor.request_openai import get_openai
from extractor.request_sonnet import get_sonnet
from extractor.request_metallama import get_meta_llama
from TabFuncFlow.utils.table_utils import markdown_to_dataframe
load_dotenv()

logger = initialize_logger(
    log_file="app_scripts.log",
    app_log_name="scripts",
    app_log_level=logging.INFO,
    log_entries={
        "extractor": logging.INFO,
    },
)


def get_llm(model: str):
    if "gpt" in model:
        return get_openai()
    elif "gemini" in model:
        return get_gemini()
    elif "sonnet" in model:
        return get_sonnet()
    elif "deepseek" in model:
        return get_deepseek()
    elif "metallama" in model:
        return get_meta_llama()
    else:
        raise ValueError(f"Invalid model: {model}")

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
 
def extract_by_csv_file(interval_time=0.0):
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pmids_fn", help="csv file path containing pmids to extract")
    parser.add_argument("-o", "--out_dir", required=True, help="output directory")
    parser.add_argument("-i", "--pmid", help="PMID(s) to extract. To specify multiple PMIDs, separate them with commas (e.g., 123456,234567,345678).")
    parser.add_argument("-m", "--model", default='gpt4o', help="model, default is gpt4o. Could be one of 'gpt4o', 'sonnet4', 'metallama4', 'gemini25flash', 'gemini20flash'.")
    args = vars(parser.parse_args())
    
    model = args.get("model", "gpt4o")
    llm = get_llm(model)
    pmid_db = get_pmid_db()
    mgr = PKPEManager(llm, pmid_db)

    pmid: str | None = args.get("pmid", None)
    pmids_fn: str | None = args.get("pmids_fn", None)
    if pmid == None and pmids_fn == None:
        print("Usage:")
        print(f"python {__file__} -o OUTPUT_FOLDER [-f PMIDS_FILE] [-i PMID] [-h]")
        return
    
    pmids = []
    if pmid is not None:
        pmids = pmid.split(',')
        pmids = [p.strip() for p in pmids]
    else:
        with open(pmids_fn, "r") as fobj:
            reader = csv.reader(fobj)
            for row in reader:
                if len(row) == 0:
                    break
                pmids.append(row[0].strip())
            
    out_dir = args["out_dir"]
    error_report = []
    for pmid in pmids:
        try:
            res = mgr.run(pmid)
            for k, value in res.items():
                value: PKPECuratedTables = value
                if not "curated_table" in value or value["curated_table"] is None:
                    logger.error(f"No curated table found for {pmid} {k}")
                    error_report.append((pmid, f"No curated table found for {pmid} {k}"))
                    continue
                df = markdown_to_dataframe(value["curated_table"])
                if df.empty:
                    continue
                out_fn = Path(out_dir) / f"{pmid}_{k}.csv"
                df.to_csv(out_fn, index=False)
                if not value["correct"]:
                    logger.error(f"Curated table for {pmid} {k} is not correct")
                    error_report.append((pmid, f"Curated table for {pmid} {k} is not correct"))
                    error_fn = Path(out_dir) / f"{pmid}_{k}_error.txt"
                    error_fn.write_text(f"Curated table for {pmid} {k} is not correct\nExplanation: {value['explanation']}\nSuggested fix: {value['suggested_fix']}\n")
                    
            time.sleep(interval_time)
        except Exception as e:
            logger.error(f"Error ocurred in curating paper {pmid}")
            logger.error(str(e))
            print(f"Error ocurred in curating paper {pmid}")
            print(str(e))
            error_report.append((pmid, str(e)))
            continue

    if len(error_report) == 0:
        logger.info("All PMIDs are successfully curated.")
        print("All PMIDs are successfully curated.")
    else:
        for err_item in error_report:
            logger.info(f"Error occurred in {err_item[0]}: {err_item[1]}")
            print(f"Error occurred in {err_item[0]}: {err_item[1]}")


if __name__ == "__main__":
    extract_by_csv_file(interval_time=5.0)
