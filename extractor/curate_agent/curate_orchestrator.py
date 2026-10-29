"""Phase 2 cascade orchestrator, per docs/VERIFICATION_AGENT_DESIGN.md.

Glues two previously-separate code paths together with a validity gate in
between, instead of building a third execution engine:

    Arm A (cheap): scripts/simple_prompt/run_simple_prompt.py, one call/table
    ValidityChecker (Tier 1 + Tier 2)
    escalate? -> Arm B (expensive): PKPEManager multi-agent mode
    ValidityChecker on Arm B too
    pick the lower error_rate; on a tie, ComparisonJudge (Tier 4, an LLM
    call) breaks it

Arm A is shelled out to exactly as documented: it is a standalone CLI script
with its own argparse-driven main() and no importable single-call entry
point. Arm B is called in-process: `PKPEManager` is already a plain Python
class, the same way `app_script_pmids.py` calls it.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pandas as pd
from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe

from extractor.agents.agent_factory import get_agent_llm, get_pipeline_llm
from extractor.agents.agent_utils import extract_pmid_info_to_db
from extractor.agents_manager.pk_pe_manager import PKPEManager
from extractor.constants import PipelineTypeEnum
from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.table_utils import format_source_tables
from extractor.curate_agent import comparison_judge
from extractor.curate_agent import validity_checker as vc

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_SIMPLE_PROMPT = os.path.join(REPO_ROOT, "scripts", "simple_prompt", "run_simple_prompt.py")

_SIMPLE_PROMPT_DIR = os.path.join(REPO_ROOT, "scripts", "simple_prompt")
if _SIMPLE_PROMPT_DIR not in sys.path:
    sys.path.insert(0, _SIMPLE_PROMPT_DIR)
import validity as _validity  # scripts/simple_prompt/validity.py

# v1 scope per the design doc's open questions: pk_summary only - label columns
# for pk_individual need confirming before that pipeline is wired in too.
DEFAULT_LABEL_COLUMNS = {
    "pk_summary": ["Analyte", "Specimen", "Population", "Parameter type"],
}

DEFAULT_PROMPT_PATHS = {
    "pk_summary": os.path.join(REPO_ROOT, "prompts", "simple_prompts", "pk_summary.md"),
}

_SCHEMA_TO_PIPELINE = {
    "pk_summary": PipelineTypeEnum.PK_SUMMARY,
}


def write_table_markdown(tables, dest_dir) -> list[str]:
    """One markdown file per source table: `<dest_dir>/table_{i}/00_markdown_table.md`.

    This is both the scratch layout `run_simple_prompt.py`'s `find_tables()`
    expects (a `table*` subdirectory per table holding `00_markdown_table.md`)
    and the Tier-2 source file list `verify_provenance`'s `--attribution` mode
    parses - so both arms, and the checker, read the exact same source text.
    """
    paths = []
    for i, table in enumerate(tables or [], start=1):
        tdir = os.path.join(dest_dir, f"table_{i}")
        os.makedirs(tdir, exist_ok=True)
        caption = table.get("caption") or ""
        footnote = table.get("footnote") or ""
        md = dataframe_to_markdown(table["table"])
        path = os.path.join(tdir, "00_markdown_table.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"{caption}\n{footnote}\n\n{md}\n")
        paths.append(path)
    return paths


def run_cheap_arm(pmid, schema, scratch_root, out_root,
                   prompt_path=None, model="qwen3.8:27b", backend="ollama",
                   temperature=0.0):
    """Arm A: one call/table via run_simple_prompt.py. Returns (csv_path, dropped_rows).

    `dropped_rows` comes straight from the manifest.json run_simple_prompt.py
    already writes - the same count docs/VERIFICATION_AGENT_DESIGN.md's Tier 1
    calls "already written to manifest.json for Arm A."
    """
    prompt_path = prompt_path or DEFAULT_PROMPT_PATHS[schema]
    cmd = [
        sys.executable, RUN_SIMPLE_PROMPT,
        "--prompt", prompt_path,
        "--schema", schema,
        "--scratch", scratch_root,
        "--out", out_root,
        "--pmids", pmid,
        "--backend", backend,
        "--model", model,
        "--temperature", str(temperature),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"run_simple_prompt.py failed (exit {result.returncode}):\n{result.stderr}"
        )

    csv_path = os.path.join(out_root, pmid, "combined.csv")
    manifest_path = os.path.join(out_root, "manifest.json")
    dropped_rows = 0
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        dropped_rows = sum(entry.get("dropped", 0) for entry in manifest
                            if entry.get("pmid") == pmid)
    return csv_path, dropped_rows


def run_expensive_arm(pmid, schema, out_dir, pmid_db):
    """Arm B: PKPEManager multi-agent mode, called in-process. Returns a csv_path.

    Model selection is the caller's environment, not a parameter here -
    PIPELINE_LLM/AGENT_LLM env vars, read by get_pipeline_llm()/get_agent_llm()
    (extractor/agents/agent_factory.py), exactly as app_script_pmids.py does.
    This arm has its own internal execution->verify->correct loop already;
    see docs/VERIFICATION_AGENT_DESIGN.md's cost note - it is not "one extra
    call."
    """
    pipeline_type = _SCHEMA_TO_PIPELINE[schema]
    mgr = PKPEManager(pipeline_llm=get_pipeline_llm(), agent_llm=get_agent_llm(),
                       pmid_db=pmid_db)
    results = mgr.run(pmid, pipeline_types=[pipeline_type])
    value = results.get(pipeline_type)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{pmid}_{schema}.csv")

    df = None
    if value and value.get("curated_table"):
        df = markdown_to_dataframe(value["curated_table"])
    if df is None or df.empty:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(_validity.schema_columns(schema))
    else:
        df.to_csv(csv_path, index=False)
    return csv_path


def curate(pmid, schema="pk_summary", html_content=None, pmid_db=None,
           work_dir=None, label_columns=None, threshold=vc.DEFAULT_THRESHOLD,
           cheap_model="qwen3.8:27b",
           run_cheap_arm_fn=run_cheap_arm, run_expensive_arm_fn=run_expensive_arm,
           compare_fn=comparison_judge.compare):
    """Run the cascade for one paper/pipeline: cheap pass, gate, maybe expensive pass, pick.

    `run_cheap_arm_fn`/`run_expensive_arm_fn`/`compare_fn` default to the real
    subprocess/PKPEManager/LLM implementations above; tests substitute fakes
    so the cascade's own logic (gate, escalation, pick-rule) can be exercised
    without a live Ollama server, multi-agent run, or LLM call.

    Returns:
        {"winner": "A" | "B", "csv_path": str, "escalated": bool,
         "candidate_a": {"csv_path": str, "report": ValidityReport},
         "candidate_b": {"csv_path": str, "report": ValidityReport} | None,
         "judge": {"winner": str, "explanation": str, "used_full_tables": bool} | None}
    """
    if schema not in DEFAULT_LABEL_COLUMNS:
        raise ValueError(
            f"schema {schema!r} is not wired for the v1 cascade "
            f"(scoped to {sorted(DEFAULT_LABEL_COLUMNS)} per "
            f"docs/VERIFICATION_AGENT_DESIGN.md)"
        )
    label_columns = label_columns or DEFAULT_LABEL_COLUMNS[schema]
    pmid_db = pmid_db or PMIDDB()
    work_dir = work_dir or os.path.join(REPO_ROOT, ".curate_agent_work", pmid)
    os.makedirs(work_dir, exist_ok=True)

    _, paper_title, paper_abstract, _, tables, _ = extract_pmid_info_to_db(
        pmid, pmid_db, html_content=html_content
    )

    scratch_root = os.path.join(work_dir, "scratch")
    source_paths = write_table_markdown(tables, os.path.join(scratch_root, pmid))

    armA_out = os.path.join(work_dir, "arm_a")
    csv_a, dropped_a = run_cheap_arm_fn(pmid, schema, scratch_root, armA_out, model=cheap_model)
    report_a = vc.check_validity(csv_a, schema, source_paths, label_columns,
                                  extra_dropped_rows=dropped_a, threshold=threshold)

    if not report_a["escalate"]:
        return {
            "winner": "A", "csv_path": csv_a, "escalated": False,
            "candidate_a": {"csv_path": csv_a, "report": report_a},
            "candidate_b": None, "judge": None,
        }

    armB_out = os.path.join(work_dir, "arm_b")
    csv_b = run_expensive_arm_fn(pmid, schema, armB_out, pmid_db)
    report_b = vc.check_validity(csv_b, schema, source_paths, label_columns,
                                  threshold=threshold)

    judge = None
    if report_a["error_rate"] < report_b["error_rate"]:
        winner, csv_path = "A", csv_a
    elif report_b["error_rate"] < report_a["error_rate"]:
        winner, csv_path = "B", csv_b
    else:
        judge = compare_fn(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            source_tables=format_source_tables(tables),
            report_a=report_a,
            table_a=dataframe_to_markdown(pd.read_csv(csv_a)),
            report_b=report_b,
            table_b=dataframe_to_markdown(pd.read_csv(csv_b)),
        )
        winner = judge["winner"]
        csv_path = csv_a if winner == "A" else csv_b

    return {
        "winner": winner, "csv_path": csv_path, "escalated": True,
        "candidate_a": {"csv_path": csv_a, "report": report_a},
        "candidate_b": {"csv_path": csv_b, "report": report_b},
        "judge": judge,
    }
