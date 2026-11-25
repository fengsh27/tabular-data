#!/usr/bin/env python
"""Phase 5 (docs/VERIFICATION_AGENT_DESIGN.md): sweep ValidityChecker's
THRESHOLD against real, already-scored benchmark data.

For each paper, two things are computed independently:

  - `error_rate_A`: ValidityChecker's Tier 1+2+3 score on Arm A's real
    curated CSV against its real source tables (threshold-independent;
    `THRESHOLD` only gates the escalate *decision*, computed here per
    candidate value instead).
  - the ground truth for "would escalating have helped": Arm A's vs Arm B's
    strict F1 against the gold table, already computed by the benchmark
    harness (`benchmark/result/.../result_f1.log`) - no need to re-run or
    re-score either arm.

Tier 3 is a live LLM call (`llm_verification.verify`) - by default this
script runs with a stub `verify_fn` that reports "always clean", so Tier 3
contributes nothing and the sweep only reflects Tier 1+2 (deterministic, no
model needed). Pass --live to use the real LLM-backed Tier 3 instead - that
needs a reachable AGENT_LLM and will make one real call per paper.

Sweeping `THRESHOLD` over `error_rate_A` and cross-referencing the F1 gap
shows the real tradeoff: how often a given threshold escalates when Arm B
genuinely was better (worth the cost) vs. escalates when Arm A was already
fine (wasted cost) vs. fails to escalate when it should have (quality loss).

Two datasets, pick with --dataset (default pk_summary_qwen36):

  pk_summary_qwen36 (9 PMIDs, collected 2026-09-25 - note this is
  `simple_qwen36`, not the `simple_qwen38` docs/VERIFICATION_AGENT_DESIGN.md
  Phase-5 example names; qwen36 is what's actually on disk for pk_summary):
    Arm A: benchmark/data/pk-summary/2026-9-25-collect-simple_qwen36/
    Arm B: benchmark/data/pk-summary/2026-9-25-collect-multiagent_gemma4/

  pk_individual_qwen38 (8 PMIDs common to both arms, collected 2026-10-01 -
  this IS the real simple-mode qwen3.8 data, but for pk_individual, not
  pk_summary - the only pipeline curate_orchestrator.py is actually wired
  for):
    Arm A: benchmark/data/pk-individual/2026-10-01-collect-simple_qwen38/
    Arm B: benchmark/data/pk-individual/2026-10-01-collect-multiagent_qwen38/
    (both arms are qwen3.8 here - NOT the qwen3.8-cheap/gemma4-expensive
    pairing the cascade actually uses, so this tests the gate's own
    signal/ground-truth alignment, not a real cost/quality tradeoff)

Source tables aren't pre-extracted anywhere, so this pulls them from
locally-cached paper HTML via the same extract_pmid_info_to_db()
+write_table_markdown() path curate_orchestrator.py uses - fully offline, no
network.

Caveats (printed again at the end of the run):
  - Small samples (8-9 papers) - a first-pass sanity check, not a
    statistically solid number.
  - error_rate_A has no `extra_dropped_rows` input here (these historical
    benchmark runs have no manifest.json), so Tier 1's structural count only
    reflects rows that reached the CSV and scored low, not rows a parser may
    have dropped outright - a lower-bound estimate, not the full picture a
    live Arm A run with a manifest would give.
  - Arm B's CSV is not scored by the checker here at all - only its
    already-computed F1 is used, as the ground-truth signal.
  - Without --live, Tier 3 contributes nothing (stubbed clean) - the sweep
    only reflects Tier 1+2.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from io import StringIO
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd

from extractor.agents.agent_utils import extract_pmid_info_to_db
from extractor.curate_agent import arms, llm_verification
from extractor.curate_agent import curate_orchestrator as co
from extractor.curate_agent import validity_checker as vc
from extractor.curate_agent.curate_orchestrator import write_table_markdown
from extractor.database.pmid_db import PMIDDB
from extractor.log_utils import initialize_logger

# Same setup app_script.py / app_script_pmids.py use: "extractor" at INFO
# routes every extractor.* logger - including llm_verification's TIER3_LLM_CALL
# records (the full Tier 3 LLM output) - to $LOGS_FOLDER (default ./logs).
logger = initialize_logger(
    log_file="calibrate_validity_threshold.log",
    app_log_name="calibrate_validity_threshold",
    app_log_level=logging.INFO,
    log_entries={"extractor": logging.INFO},
)

PK_SUMMARY_QWEN36_PMIDS = [
    "16143486", "17635501", "22050870", "29943508", "30825333",
    "34114632", "34183327", "35465728", "35489632",
]

PK_INDIVIDUAL_QWEN38_PMIDS = [
    "10971311", "11849190", "18426260", "23200982", "24989434",
    "32056930", "32153014", "34746508",  # 33253437 has no multiagent_qwen38 output
]

# All 9 pk_individual gold papers; the multiagent_gemma4 collection covers every one.
PK_INDIVIDUAL_ALL_PMIDS = PK_INDIVIDUAL_QWEN38_PMIDS[:7] + ["33253437", "34746508"]

DATASETS = {
    "pk_summary_qwen36": {
        "schema": "pk_summary",
        "pmids": PK_SUMMARY_QWEN36_PMIDS,
        "html_dir": os.path.join(REPO_ROOT, "data"),
        "arm_a_dir": os.path.join(REPO_ROOT, "benchmark/data/pk-summary/2026-9-25-collect-simple_qwen36"),
        "arm_a_suffix": "_qwen36.csv",
        "cheap_model": "qwen3.6:35b",  # Ollama tag of the model that produced arm_a_dir (--per-table re-runs Arm A live)
        "gold_dir": os.path.join(REPO_ROOT, "benchmark/data/pk-summary/baseline"),
        "arm_a_f1_log": os.path.join(
            REPO_ROOT,
            "benchmark/result/pk-summary/2026-9-25-collect-simple_qwen36-baseline/result_f1.log",
        ),
        "arm_b_f1_log": os.path.join(
            REPO_ROOT,
            "benchmark/result/pk-summary/2026-9-25-collect-multiagent_gemma4-baseline/result_f1.log",
        ),
    },
    "pk_individual_qwen38": {
        "schema": "pk_individual",
        "pmids": PK_INDIVIDUAL_QWEN38_PMIDS,
        "html_dir": os.path.join(REPO_ROOT, "benchmark/data/source/pk-individual"),
        "arm_a_dir": os.path.join(REPO_ROOT, "benchmark/data/pk-individual/2026-10-01-collect-simple_qwen38"),
        "arm_a_suffix": "_qwen38.csv",
        "cheap_model": "qwen3.8:27b",
        "gold_dir": os.path.join(REPO_ROOT, "benchmark/data/pk-individual/baseline"),
        "arm_a_f1_log": os.path.join(
            REPO_ROOT,
            "benchmark/result/pk-individual/2026-10-01-collect-simple_qwen38-baseline/result_f1.log",
        ),
        "arm_b_f1_log": os.path.join(
            REPO_ROOT,
            "benchmark/result/pk-individual/2026-10-01-collect-multiagent_qwen38-baseline/result_f1.log",
        ),
    },
    # Round 4 (2026-10-31): Arm A = qwen3.8:27b simple prompt, Arm B / Tier 3 / judge = QWEN3.8-27B,
    # all 9 gold papers. The pipeline_qwen38 and multiagent_qwen38 collections both lack 33253437,
    # so its F1_B is blank; the Slurm script picks the matching log with --arm-b-f1-log.
    "pk_individual_qwen38_all": {
        "schema": "pk_individual",
        "pmids": PK_INDIVIDUAL_ALL_PMIDS,
        "html_dir": os.path.join(REPO_ROOT, "benchmark/data/source/pk-individual"),
        "arm_a_dir": os.path.join(REPO_ROOT, "benchmark/data/pk-individual/2026-10-01-collect-simple_qwen38"),
        "arm_a_suffix": "_qwen38.csv",
        "cheap_model": "qwen3.8:27b",
        "gold_dir": os.path.join(REPO_ROOT, "benchmark/data/pk-individual/baseline"),
        "arm_a_f1_log": os.path.join(
            REPO_ROOT,
            "benchmark/result/pk-individual/2026-10-01-collect-simple_qwen38-baseline/result_f1.log",
        ),
        "arm_b_f1_log": os.path.join(
            REPO_ROOT,
            "benchmark/result/pk-individual/2026-10-01-collect-pipeline_qwen38-baseline/result_f1.log",
        ),
    },
    # pk_summary round 6 (2026-10-31): Arm A = qwen3.8:27b simple prompt, Arm B / Tier 3 / judge =
    # QWEN3.8-27B, the 9 pk_summary gold papers with source HTML (30950674 has none). No stored
    # qwen3.8 simple/pipeline pk-summary collection exists, so there are no reference F1 logs;
    # run_cascade_benchmark scores both arms from the run itself anyway.
    "pk_summary_qwen38": {
        "schema": "pk_summary",
        "pmids": PK_SUMMARY_QWEN36_PMIDS,
        "html_dir": os.path.join(REPO_ROOT, "data"),
        "arm_a_dir": None,
        "arm_a_suffix": None,
        "cheap_model": "qwen3.8:27b",
        "gold_dir": os.path.join(REPO_ROOT, "benchmark/data/pk-summary/baseline"),
        "arm_a_f1_log": None,
        "arm_b_f1_log": None,
    },
    # Round 3 (2026-10-31): Arm A unchanged (qwen3.8:27b simple prompt); Arm B, Tier 3 and the
    # judge run on GEMMA4-31B (AGENT_LLM/PIPELINE_LLM set by the Slurm script), so F1_B is the
    # multiagent_gemma4 collection and all 9 gold papers are in scope.
    "pk_individual_qwen38_gemma4": {
        "schema": "pk_individual",
        "pmids": PK_INDIVIDUAL_ALL_PMIDS,
        "html_dir": os.path.join(REPO_ROOT, "benchmark/data/source/pk-individual"),
        "arm_a_dir": os.path.join(REPO_ROOT, "benchmark/data/pk-individual/2026-10-01-collect-simple_qwen38"),
        "arm_a_suffix": "_qwen38.csv",
        "cheap_model": "qwen3.8:27b",
        "gold_dir": os.path.join(REPO_ROOT, "benchmark/data/pk-individual/baseline"),
        "arm_a_f1_log": os.path.join(
            REPO_ROOT,
            "benchmark/result/pk-individual/2026-10-01-collect-simple_qwen38-baseline/result_f1.log",
        ),
        "arm_b_f1_log": os.path.join(
            REPO_ROOT,
            "benchmark/result/pk-individual/2026-9-22-collect-multiagent_gemma4-baseline/result_f1.log",
        ),
    },
}

# --- the eight other pipelines, on the pk-pe benchmark papers -----------------
#
# No manual gold exists for these pipelines. The pk-pe benchmark
# (benchmark/test_pk_pe_benchmark_with_semantic.py) compares two model
# runs instead - 2026-4-10 (qwen3.5) against 2026-4-16 (gpt-4o, the
# "baseline" argument). The cascade datasets below therefore carry no
# gold_dir; `reference` names the gpt-4o set so run_cascade_benchmark.py can
# print an F1 *against gpt-4o* (labelled as such), and the per-paper reports
# (escalations, Tier 3 verdicts, judge picks) are the main output. The
# papers' source files are the PMC XML staged for the skills route benchmark
# (HtmlTableExtractor parses XML too); each schema's paper list is the union
# of the papers that have that pipeline in either benchmark version.
PKPE_SOURCE_DIR = "/fs/scratch/PCON0100/feng1426/projects/playground/tabular-data-skills-route/.staging"
PKPE_REFERENCE_DIR = os.path.join(REPO_ROOT, "benchmark/data/pk-pe/2026-4-16")
PKPE_PMIDS_BY_SCHEMA = {
    "pk_drug_summary": ["18391836", "18665197", "18673410", "18686186", "18694901", "18701886",
                        "18758393", "18762498"],
    "pk_drug_individual": ["18701886"],
    "pk_specimen_summary": ["18665197", "18686186", "18694901", "18701886", "18758393", "18762498"],
    "pk_specimen_individual": ["18701886"],
    "pk_population_summary": ["18665197", "18686186", "18694901", "18762498"],
    "pk_population_individual": [],  # no paper of the pk-pe set ran this pipeline; add PMIDs to try it
    "pe_study_info": ["18670224"],
    "pe_study_outcome": ["18670224"],
}
PKPE_ALL_PMIDS = ["18391836", "18665197", "18670224", "18673410", "18686186", "18694901",
                  "18701886", "18758393", "18762498"]
for _schema, _pmids in PKPE_PMIDS_BY_SCHEMA.items():
    DATASETS[f"pkpe_{_schema}_qwen38"] = {
        "schema": _schema,
        "pmids": list(_pmids),
        "html_dir": PKPE_SOURCE_DIR,
        "source_ext": ".xml",
        "cheap_model": "qwen3.8:27b",
        "reference": {"dir": PKPE_REFERENCE_DIR, "suffix": f"_{_schema}_gpt4o.csv", "label": "gpt4o"},
    }
    # the same schema on every pk-pe paper, for papers the route step would not have picked
    DATASETS[f"pkpe_{_schema}_qwen38_all"] = {**DATASETS[f"pkpe_{_schema}_qwen38"], "pmids": list(PKPE_ALL_PMIDS)}

DEFAULT_THRESHOLDS = [round(i * 0.05, 2) for i in range(21)]  # 0.00 .. 1.00
DEFAULT_EPSILON = 0.05  # minimum F1 gap to call escalating "worth it"


def _stub_verify(**kwargs):
    """Default Tier 3: always clean, no LLM call. Pass --live for the real one."""
    return {"correct": True, "error_row_count": 0, "missed_row_count": 0,
            "explanation": "All values match."}


def load_strict_f1(log_path: str, pmids: list[str]) -> dict[str, float]:
    """pmid -> strict-mode F1, from a result_f1.log (skips MICRO/MACRO rows)."""
    f1 = {}
    with open(log_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            row = [c.strip() for c in row]
            if len(row) < 6:
                continue
            _model, pmid, mode, _precision, _recall, f1_value = row[:6]
            if mode == "strict" and pmid in pmids:
                f1[pmid] = float(f1_value)
    return f1


def read_source(pmid: str, html_dir: str, ext: str | None = None) -> str:
    """The paper's source file: `<html_dir>/<pmid><ext>`; with `ext` None, .html
    then .xml (HtmlTableExtractor parses both)."""
    exts = [ext] if ext else [".html", ".xml"]
    for e in exts:
        path = os.path.join(html_dir, f"{pmid}{e}")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return fh.read()
    raise FileNotFoundError(f"no source for {pmid} in {html_dir} ({', '.join(exts)})")


def extract_paper_tables(pmid: str, html_dir: str, pmid_db: PMIDDB, scratch_dir: str):
    """(paper_title, paper_abstract, tables, source_paths) for `pmid`, from cached HTML.

    extract_pmid_info_to_db()'s first-ever call for a pmid returns tables
    whose "table" field is a JSON string, not a DataFrame - insert_pmid_info()
    serializes it in place as a storage side effect before handing the same
    list back (see extractor/database/pmid_db.py:95-98). select_pmid_info()
    (the path a second call for the same pmid takes) converts it back; this
    normalizes it the same way for the first call too, rather than touching
    that shared helper for a one-off calibration script.
    """
    html_content = read_source(pmid, html_dir)
    _, paper_title, paper_abstract, _, tables, _ = extract_pmid_info_to_db(
        pmid, pmid_db, html_content=html_content)
    for table in tables:
        if isinstance(table["table"], str):
            table["table"] = pd.read_json(StringIO(table["table"]))
    source_paths = write_table_markdown(tables, os.path.join(scratch_dir, pmid))
    return paper_title, paper_abstract, tables, source_paths


def extract_paper(pmid: str, html_dir: str, pmid_db: PMIDDB, scratch_dir: str):
    """(paper_title, paper_abstract, source_paths) - see extract_paper_tables."""
    paper_title, paper_abstract, _, source_paths = extract_paper_tables(pmid, html_dir, pmid_db, scratch_dir)
    return paper_title, paper_abstract, source_paths


def compute_per_table_reports(pmid: str, dataset: dict, pmid_db: PMIDDB, scratch_dir: str,
                              work_dir: str, verify_fn, cheap_model: str) -> list[dict]:
    """The v2 cascade's gate inputs, one per selected table: Arm A re-run live
    on that table alone (arms.run_cheap_arm_table), then Tier 1+2+3 against
    that table's own source. Returns [{"table_index", "report", "seconds"}].
    No Arm B - this measures where the gate would fire, not what it would pick."""
    schema = dataset["schema"]
    paper_title, paper_abstract, tables, source_paths = extract_paper_tables(
        pmid, dataset["html_dir"], pmid_db, scratch_dir)
    positions = co.select_tables(tables=tables, schema=schema)
    out = []
    for pos in positions:
        table_dir = os.path.join(work_dir, "per_table", pmid, schema, f"table_{pos + 1}")
        a = arms.run_cheap_arm_table(pmid, schema, source_paths[pos], os.path.join(table_dir, "arm_a"),
                                     model=cheap_model)
        report = vc.check_validity(a["csv_path"], schema, [source_paths[pos]], paper_title, paper_abstract,
                                   extra_dropped_rows=a["dropped_rows"], verify_fn=verify_fn)
        out.append({"table_index": pos + 1, "report": report, "seconds": a["seconds"]})
    return out


def compute_report(pmid: str, dataset: dict, pmid_db: PMIDDB, scratch_dir: str, verify_fn) -> dict:
    csv_path = os.path.join(dataset["arm_a_dir"], f"{pmid}{dataset['arm_a_suffix']}")
    paper_title, paper_abstract, source_paths = extract_paper(
        pmid, dataset["html_dir"], pmid_db, scratch_dir)
    return vc.check_validity(
        csv_path, dataset["schema"], source_paths, paper_title, paper_abstract,
        verify_fn=verify_fn,
    )


def classify(error_rate: float, threshold: float, f1_a: float, f1_b: float, epsilon: float) -> str:
    would_escalate = error_rate > threshold
    helps = (f1_b - f1_a) > epsilon
    if would_escalate and helps:
        return "escalated_correctly"
    if would_escalate and not helps:
        return "escalated_wastefully"
    if not would_escalate and helps:
        return "missed_escalation"
    return "stayed_correctly"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="pk_summary_qwen36",
                     help="which benchmark dataset to sweep (default: pk_summary_qwen36)")
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                     help=f"min F1 gap (B - A) to call escalating worth it (default {DEFAULT_EPSILON})")
    ap.add_argument("--thresholds", default=None,
                     help="comma-separated THRESHOLD values to sweep (default: 0.00..1.00 step 0.05)")
    ap.add_argument("--work-dir", default=None,
                     help="scratch dir for the pmid_info.db cache + extracted source tables "
                          "(default: .curate_agent_calibration/<dataset>)")
    ap.add_argument("--live", action="store_true",
                     help="use the real LLM-backed Tier 3 instead of the always-clean stub "
                          "(needs a reachable AGENT_LLM; makes one real call per paper)")
    ap.add_argument("--pmid", default=None,
                     help="restrict the sweep to this one PMID from the dataset "
                          "(e.g. for a one-paper-per-Slurm-job run)")
    ap.add_argument("--per-table", action="store_true",
                     help="v2 cascade gate: select tables (1 LLM call), re-run Arm A live on each "
                          "selected table and check it against its own source; one line per "
                          "table, then the sweep on the paper's worst table. Requires --live.")
    ap.add_argument("--cheap-model", default=None,
                     help="Ollama tag for --per-table's Arm A (default: the dataset's own cheap_model)")
    args = ap.parse_args(argv)
    if args.per_table and not args.live:
        raise SystemExit("--per-table re-runs Arm A and the table selector live; pass --live too")

    dataset = DATASETS[args.dataset]
    work_dir = args.work_dir or os.path.join(REPO_ROOT, ".curate_agent_calibration", args.dataset)
    thresholds = (
        [float(t) for t in args.thresholds.split(",")] if args.thresholds else DEFAULT_THRESHOLDS
    )
    verify_fn = llm_verification.verify if args.live else _stub_verify

    os.makedirs(work_dir, exist_ok=True)
    pmid_db = PMIDDB(Path(work_dir) / "pmid_info.db")
    scratch_dir = os.path.join(work_dir, "scratch")

    pmids = dataset["pmids"]
    if args.pmid is not None:
        if args.pmid not in pmids:
            raise SystemExit(f"--pmid {args.pmid} is not in dataset {args.dataset!r}'s PMIDs: {pmids}")
        pmids = [args.pmid]
    f1_a_all = load_strict_f1(dataset["arm_a_f1_log"], pmids)
    f1_b_all = load_strict_f1(dataset["arm_b_f1_log"], pmids)

    print(f"Dataset: {args.dataset} (schema={dataset['schema']}, {len(pmids)} PMIDs, "
          f"Tier 3 {'LIVE' if args.live else 'stubbed clean'})\n")

    def _tier_cells(report):
        total_rows = report["tier1"]["total_rows"]
        # Denominator matches error_rate's own Tier 3 rate: total_rows +
        # missed_row_count, not just total_rows - see validity_checker.py.
        tier3 = (f"{report['tier3']['error_row_count'] + report['tier3']['missed_row_count']}"
                 f"/{total_rows + report['tier3']['missed_row_count']}"
                 if report["tier3"] else "skipped")
        return f"{report['tier1']['dropped_rows']}/{total_rows}", tier3

    rows = []
    if args.per_table:
        cheap_model = args.cheap_model or dataset["cheap_model"]
        print(f"Per-table gate (Arm A model {cheap_model}); escalates at the default threshold "
              f"{vc.DEFAULT_THRESHOLD} when error_A > threshold:\n")
        print(f"{'pmid':<10} {'table':>5} {'error_A':>8} {'tier1(d/n)':>11} {'tier3(f/n)':>10} "
              f"{'armA_s':>7} {'escalate':>8}")
        for pmid in pmids:
            per_table = compute_per_table_reports(pmid, dataset, pmid_db, scratch_dir, work_dir,
                                                  verify_fn=verify_fn, cheap_model=cheap_model)
            for t in per_table:
                tier1, tier3 = _tier_cells(t["report"])
                print(f"{pmid:<10} {t['table_index']:>5} {t['report']['error_rate']:>8.3f} {tier1:>11} "
                      f"{tier3:>10} {t['seconds']:>7.0f} {'yes' if t['report']['escalate'] else 'no':>8}")
            # Paper-level stand-in for the sweep: the paper escalates if ANY table does,
            # so its gate value is the worst table's error_rate.
            worst = max((t["report"]["error_rate"] for t in per_table), default=0.0)
            rows.append({
                "pmid": pmid, "error_rate_a": worst,
                "tier1": f"{len(per_table)} tables", "tier3": "see above",
                "f1_a": f1_a_all[pmid], "f1_b": f1_b_all[pmid],
                "f1_gap": f1_b_all[pmid] - f1_a_all[pmid],
            })
        print("\nPaper roll-up (error_A = worst table):")
    else:
        for pmid in pmids:
            report = compute_report(pmid, dataset, pmid_db, scratch_dir, verify_fn=verify_fn)
            tier1, tier3 = _tier_cells(report)
            rows.append({
                "pmid": pmid, "error_rate_a": report["error_rate"], "tier1": tier1, "tier3": tier3,
                "f1_a": f1_a_all[pmid], "f1_b": f1_b_all[pmid],
                "f1_gap": f1_b_all[pmid] - f1_a_all[pmid],
            })

    print(f"{'pmid':<10} {'error_A':>8} {'tier1(d/n)':>11} "
          f"{'tier3(f/n)':>10} {'F1_A':>6} {'F1_B':>6} {'gap':>6}")
    for r in rows:
        print(f"{r['pmid']:<10} {r['error_rate_a']:>8.3f} {r['tier1']:>11} "
              f"{r['tier3']:>10} {r['f1_a']:>6.3f} {r['f1_b']:>6.3f} {r['f1_gap']:>+6.3f}")

    print(f"\nSweeping THRESHOLD (escalate if error_rate_A > THRESHOLD), "
          f"epsilon={args.epsilon} (min F1 gap to call escalation worth it):\n")
    print(f"{'THRESHOLD':>10} {'correct_esc':>12} {'wasted_esc':>11} "
          f"{'missed_esc':>11} {'correct_stay':>13}")
    for t in thresholds:
        counts = {"escalated_correctly": 0, "escalated_wastefully": 0,
                  "missed_escalation": 0, "stayed_correctly": 0}
        for r in rows:
            outcome = classify(r["error_rate_a"], t, r["f1_a"], r["f1_b"], args.epsilon)
            counts[outcome] += 1
        print(f"{t:>10.2f} {counts['escalated_correctly']:>12} {counts['escalated_wastefully']:>11} "
              f"{counts['missed_escalation']:>11} {counts['stayed_correctly']:>13}")

    print(
        f"\nCaveats: {len(pmids)} papers only (sanity check, not a statistically solid "
        "number); error_rate_A has no extra_dropped_rows input (no manifest.json for "
        "this historical run, so Tier 1's count is a lower bound); Arm B's CSV isn't "
        "scored by the checker here at all - only its already-computed F1 is used as "
        "ground truth." + ("" if args.live else " Tier 3 is stubbed clean (pass --live "
        "for the real LLM-backed check) - this sweep only reflects Tier 1+2.")
    )


if __name__ == "__main__":
    main()
