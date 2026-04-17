"""
python app_script_pmids.py -o OUTPUT_FOLDER [-f PMIDS_FILE] [-i PMID] [-j JOB_ID] [-p PIPELINES] [-h]

This script will:
1. Read PMIDs from a CSV file or a single PMID
2. Extract tables from the HTML content of each PMID
3. Insert the extracted data into a SQLite database
4. Run PK-PE agents to curate PK-PE information from the extracted data
5. Write the curated data into CSV files in the output directory
6. Write a per-pipeline summary (pmid, pipeline, final_answer, suggested_fix) incrementally

Input:
- optional -f  CSV file (PMID, HTML_FILE_PATH) or (PMID) columns
- optional -i  single PMID
- optional -o  output directory (skips agent step if omitted)
- optional -j  job ID for the summary filename (defaults to PID)
- optional -p  space-separated pipeline names to force for every PMID, skipping the
               design step. Valid values: pk_summary, pk_individual,
               pk_specimen_summary, pk_specimen_individual, pk_drug_summary,
               pk_drug_individual, pk_population_summary, pk_population_individual,
               pe_study_info, pe_study_outcome
"""

import argparse
import contextlib
import csv
import logging
import os
from pathlib import Path
import time

from dotenv import load_dotenv

from extractor.agents.agent_factory import get_agent_llm, get_pipeline_llm
from extractor.agents.agent_utils import extract_pmid_info_to_db
from extractor.agents.pk_pe_agents.pk_pe_agents_types import FinalAnswerEnum, PKPECuratedTables
from extractor.agents_manager.pk_pe_manager import PKPEManager
from extractor.constants import PipelineTypeEnum
from extractor.database.pmid_db import PMIDDB
from extractor.log_utils import initialize_logger
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.pmid_extractor.pubmed_fulltext import PubMedFullTextRetriever
from extractor.utils import (
    convert_html_to_text_no_table,
    convert_sections_to_full_text,
    remove_references,
)
from TabFuncFlow.utils.table_utils import markdown_to_dataframe

load_dotenv()

logger = initialize_logger(
    log_file="app_scripts.log",
    app_log_name="scripts",
    app_log_level=logging.INFO,
    log_entries={"extractor": logging.INFO},
)

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _get_logs_folder() -> Path:
    folder = os.environ.get("LOGS_FOLDER", "").strip() or "./logs"
    return Path(folder)


def _create_pmid_log_handler(pmid: str, logs_folder: Path) -> logging.FileHandler:
    handler = logging.FileHandler(logs_folder / f"{pmid}.log", mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


@contextlib.contextmanager
def _pmid_log_context(pmid: str, logs_folder: Path):
    handler = _create_pmid_log_handler(pmid, logs_folder)
    target_loggers = [logger, logging.getLogger("extractor")]
    for lg in target_loggers:
        lg.addHandler(handler)
    try:
        yield
    finally:
        for lg in target_loggers:
            lg.removeHandler(handler)
        handler.close()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_pmid_db() -> PMIDDB:
    db_dir = Path(os.environ.get("DATA_FOLDER", "./data")) / "databases"
    os.makedirs(db_dir, exist_ok=True)
    return PMIDDB(db_dir / "pmid_info.db")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data_by_pmids_csv_file(csv_pmids_fn: str, pmid_db: PMIDDB) -> list[str]:
    csv_path = Path(csv_pmids_fn)
    base_dir = csv_path.parent
    extractor = HtmlTableExtractor()
    fulltext_retriever = PubMedFullTextRetriever()
    inserted = skipped = failed = 0
    pmids = []

    with open(csv_pmids_fn, "r", encoding="utf-8") as fobj:
        reader = csv.reader(fobj)
        for row_idx, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue

            pmid = row[0].strip()
            html_path = row[1].strip() if len(row) > 1 else ""

            if row_idx == 1 and pmid.lower() in {"pmid", "pmcid"}:
                continue
            if not pmid:
                logger.warning(f"Row {row_idx}: missing pmid. Skipping.")
                skipped += 1
                continue
            if pmid_db.select_pmid_info(pmid) is not None:
                pmids.append(pmid)
                logger.info(f"PMID {pmid} already exists in DB. Skipping.")
                skipped += 1
                continue

            if html_path:
                html_file = Path(html_path)
                if not html_file.is_absolute():
                    html_file = (base_dir / html_file).resolve()

                try:
                    html_content = html_file.read_text(encoding="utf-8", errors="ignore")
                except OSError as e:
                    logger.error(f"Row {row_idx}: failed to read {html_file}: {e}")
                    failed += 1
                    continue
            else:
                try:
                    result = fulltext_retriever.retrieve(pmid, fallback=True)
                except Exception as e:
                    logger.error(
                        f"Row {row_idx}: failed to retrieve full text for PMID {pmid}: {e}"
                    )
                    failed += 1
                    continue

                if result.code >= 400 or not result.content:
                    logger.error(
                        f"Row {row_idx}: full text retrieval for PMID {pmid} "
                        f"returned code {result.code}"
                    )
                    failed += 1
                    continue

                if (result.content_type or "").split(";")[0].strip() != "text/html" and \
                    (result.content_type or "").split(";")[0].strip() != "application/xml":
                    logger.error(
                        f"Row {row_idx}: unsupported content type "
                        f"'{result.content_type}' for PMID {pmid}"
                    )
                    failed += 1
                    continue

                html_content = (
                    result.content
                    if isinstance(result.content, str)
                    else result.content.decode("utf-8", errors="ignore")
                )

            try:
                tables = extractor.extract_tables(html_content) or []
                sections = extractor.extract_sections(html_content) or []
                abstract = extractor.extract_abstract(html_content)
                title = extractor.extract_title(html_content)
                full_text = (
                    convert_sections_to_full_text(sections)
                    if sections
                    else remove_references(convert_html_to_text_no_table(html_content))
                )
                ok = pmid_db.insert_pmid_info(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    full_text=full_text,
                    tables=tables,
                    sections=sections,
                )
                if ok:
                    inserted += 1
                    pmids.append(pmid)
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Row {row_idx}: failed to parse HTML for PMID {pmid}: {e}")
                failed += 1

    logger.info(
        f"prepare_data_by_pmids_csv_file completed: "
        f"inserted={inserted}, skipped={skipped}, failed={failed}"
    )
    return pmids


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------

def _parse_pipelines(names: list[str]) -> list[PipelineTypeEnum]:
    """Validate and convert pipeline name strings to PipelineTypeEnum values."""
    valid = {m.value: m for m in PipelineTypeEnum}
    result = []
    errors = []
    for name in names:
        if name in valid:
            result.append(valid[name])
        else:
            errors.append(name)
    if errors:
        print(f"Invalid pipeline name(s): {errors}")
        print(f"Valid options: {list(valid.keys())}")
        raise SystemExit(1)
    return result


def _curate_pmid(
    pmid: str,
    mgr: PKPEManager,
    out_dir: Path,
    write_summary,
    pipeline_types: list[PipelineTypeEnum] | None = None,
) -> list[tuple[str, str]]:
    """Run all pipelines for one PMID. Returns a list of (pmid, error_msg) tuples."""
    errors = []
    try:
        res = mgr.run(pmid, pipeline_types=pipeline_types)
    except Exception as e:
        logger.error(f"Identification/design step failed for {pmid}: {e}")
        write_summary([pmid, "N/A", "Error", "N/A"])
        return [(pmid, str(e))]

    if not res:
        logger.info(f"Paper {pmid} identified as Neither PK nor PE, skipping curation.")
        write_summary([pmid, "N/A", "Neither", "N/A"])
        return []

    for pipeline_type, value in res.items():
        pipeline_type: PipelineTypeEnum
        value: PKPECuratedTables

        final_answer = (
            value["correct"].value if hasattr(value["correct"], "value") else str(value["correct"])
        )
        suggested_fix = value.get("suggested_fix") or "N/A"
        write_summary([pmid, pipeline_type.value, final_answer, suggested_fix])

        if not value.get("curated_table"):
            msg = f"No curated table found for {pmid} {pipeline_type.value}"
            logger.error(msg)
            errors.append((pmid, msg))
            continue

        df = markdown_to_dataframe(value["curated_table"])
        if df.empty:
            continue

        df.to_csv(out_dir / f"{pmid}_{pipeline_type.value}.csv", index=False)

        if value["correct"] != FinalAnswerEnum.Correct:
            msg = f"Curated table for {pmid} {pipeline_type.value} is not correct"
            logger.error(msg)
            errors.append((pmid, msg))
            (out_dir / f"{pmid}_{pipeline_type.value}_error.txt").write_text(
                f"{msg}\nExplanation: {value['explanation']}\nSuggested fix: {value['suggested_fix']}\n",
                encoding="utf-8",
            )

    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract_by_csv_file(interval_time: float = 0.0):
    parser = argparse.ArgumentParser(
        description="Curate PK-PE data from papers identified by PMID."
    )
    parser.add_argument("-f", "--pmids_fn", help="CSV file path containing PMIDs to extract")
    parser.add_argument("-i", "--pmid", help="Single paper PMID")
    parser.add_argument("-o", "--out_dir", help="Output directory")
    parser.add_argument(
        "-j", "--job_id",
        help="Job ID for the summary filename (defaults to PID)",
        type=int,
    )
    parser.add_argument(
        "-p", "--pipelines",
        nargs="+",
        metavar="PIPELINE",
        help=(
            "Space-separated pipeline names to run for every PMID, skipping the "
            "design step. E.g.: -p pk_summary pk_individual"
        ),
    )
    args = vars(parser.parse_args())

    pmids_fn: str | None = args.get("pmids_fn")
    pmid: str | None = args.get("pmid")
    out_dir: str | None = args.get("out_dir")
    job_id: int = args.get("job_id") or os.getpid()
    pipeline_names: list[str] | None = args.get("pipelines")
    pipeline_types: list[PipelineTypeEnum] | None = (
        _parse_pipelines(pipeline_names) if pipeline_names else None
    )

    if pmids_fn is None and pmid is None:
        parser.print_help()
        return

    pmid_db = get_pmid_db()
    if pmids_fn is not None:
        pmids = prepare_data_by_pmids_csv_file(pmids_fn, pmid_db)
    else:
        extract_pmid_info_to_db(pmid, pmid_db)
        pmids = [pmid]

    if not pmids:
        logger.info("No PMIDs to process")
        return

    if out_dir is None:
        logger.info("No output directory provided, skipping agent part")
        return

    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)

    logs_folder = _get_logs_folder()
    os.makedirs(logs_folder, exist_ok=True)

    input_stem = Path(pmids_fn).stem if pmids_fn is not None else pmid
    summary_fn = out_path / f"summary_{job_id}_{input_stem}.csv"

    mgr = PKPEManager(
        pipeline_llm=get_pipeline_llm(),
        agent_llm=get_agent_llm(),
        pmid_db=pmid_db,
    )

    error_report = []
    with open(summary_fn, "w", newline="", encoding="utf-8") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(["pmid", "pipeline", "final_answer", "suggested_fix"])

        def write_summary(row):
            summary_writer.writerow(row)
            summary_file.flush()

        for pmid in pmids:
            with _pmid_log_context(pmid, logs_folder):
                try:
                    logger.info(f"Start curating paper {pmid}")
                    errors = _curate_pmid(pmid, mgr, out_path, write_summary, pipeline_types)
                    error_report.extend(errors)
                    logger.info(f"Finish curating paper {pmid}")
                except Exception as e:
                    logger.error(f"Error occurred in curating paper {pmid}: {e}")
                    write_summary([pmid, "N/A", "Error", "N/A"])
                    error_report.append((pmid, str(e)))
                time.sleep(interval_time)

    logger.info(f"Summary written to {summary_fn}")

    if not error_report:
        logger.info("All PMIDs are successfully curated.")
        print("All PMIDs are successfully curated.")
    else:
        for pmid, msg in error_report:
            logger.error(f"Error occurred in {pmid}: {msg}")
            print(f"Error occurred in {pmid}: {msg}")


if __name__ == "__main__":
    extract_by_csv_file(interval_time=5.0)
