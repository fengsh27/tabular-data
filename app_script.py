import argparse
import csv
import logging
import os
import time
from dotenv import load_dotenv

from extractor.log_utils import initialize_logger
from extractor.pk_sum_extractor import extract_pk_summary
from extractor.request_deepseek import get_deepseek
from extractor.request_geminiai import get_gemini
from extractor.request_openai import get_openai

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
    return (
        get_openai()
        if model.startswith("gpt")
        else (get_deepseek() if model.startswith("deepseek") else get_gemini())
    )


def extract_pk_summary_by_csv_file(interval_time=0.0):
    model = "gpt4o"
    llm = get_llm(model)
    parser = argparse.ArgumentParser()
    parser.add_argument("pmids_fn", help="csv file path containing pmids to extract")
    parser.add_argument("out_dir", help="output directory")
    args = vars(parser.parse_args())

    out_dir = args["out_dir"]
    error_report = []
    with open(args["pmids_fn"]) as fobj:
        reader = csv.reader(fobj)
        for row in reader:
            if len(row) == 0:
                break
            pmid = row[0].strip()
            try:
                res, error_msg, df, token_usage = extract_pk_summary(pmid, llm)
                if not res or df is None:
                    logger.error(f"Error occurred in curating paper {pmid}")
                    logger.error(error_msg)
                    print(f"Error occurred in curating paper {pmid}")
                    print(error_msg)
                    error_report.append((pmid, error_msg))
                    time.sleep(interval_time)
                    continue
                df.to_csv(os.path.join(out_dir, pmid + f"_{model}.csv"))
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
    extract_pk_summary_by_csv_file(interval_time=5.0)
