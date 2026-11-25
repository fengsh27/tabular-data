#!/usr/bin/env python
"""Run the v2 per-table CurateAgent cascade LIVE on one of the calibration
datasets and score its combined output against gold.

    python scripts/run_cascade_benchmark.py --dataset pk_summary_qwen36 [--pmid 16143486]

Per paper: curate_orchestrator.curate(pmid, schemas=[schema]) with the real
arms (Arm A = run_simple_prompt in-process on the dataset's cheap_model; Arm B
= the multi-agent workflow via PIPELINE_LLM/AGENT_LLM; Tier 3 and the judge via
AGENT_LLM), then strict F1 of <work_dir>/cascade/<pmid>/<schema>/combined.csv
against benchmark/data/<pipeline>/baseline/<pmid>_baseline.csv with the same
TablesF1Evaluator + preprocess the benchmark suite uses.

F1_A and F1_B come from THIS run, never from a stored collection:
  F1_A  the paper rebuilt from Arm A's table for every selected table (Arm A
        runs on every table, so this is the full "Arm A alone" score);
  F1_B  the paper rebuilt from Arm B's table for every escalated table and
        Arm A's for the rest - what the cascade would have scored had the
        judge picked B everywhere it ran. Arm B only runs on escalated
        tables, so this is n/a for a paper with no escalation.
`--reference-f1` adds the dataset's historical result_f1.log numbers
(F1_A_ref / F1_B_ref) as extra columns, for comparison only when asked.

Writes <work_dir>/cascade_summary[_<pmid>].json. Needs OLLAMA_BASE_URL,
AGENT_LLM, PIPELINE_LLM in the environment (see scripts/slurm/slurm_cascade_v2_*_array.sh).
Logging goes through extractor/log_utils.py (LOGS_FOLDER) via the
calibration module's initialize_logger().
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.configs import (  # noqa: E402
    PK_INDIVIDUAL_ANCHOR_COLUMNS, PK_INDIVIDUAL_COLUMNS_TYPE, PK_INDIVIDUAL_RATING_COLUMNS,
    PK_SUMMARY_ANCHOR_COLUMNS, PK_SUMMARY_COLUMNS_TYPE, PK_SUMMARY_RATING_COLUMNS,
)
from benchmark.pk_preprocess import preprocess_pk_individual_table, preprocess_pk_summary_table  # noqa: E402
from benchmark.configs import get_benchmark_config  # noqa: E402
from benchmark.constant import BenchmarkType  # noqa: E402
from benchmark.test_f1_with_semantic import TablesF1Evaluator  # noqa: E402
from extractor.curate_agent import curate_orchestrator as co  # noqa: E402
from extractor.database.pmid_db import PMIDDB  # noqa: E402
from scripts.calibrate_validity_threshold import DATASETS, load_strict_f1, read_source  # noqa: E402

SCORING = {
    "pk_summary": dict(preprocess=preprocess_pk_summary_table, rating_cols=PK_SUMMARY_RATING_COLUMNS,
                       anchor_cols=PK_SUMMARY_ANCHOR_COLUMNS, columns_type=PK_SUMMARY_COLUMNS_TYPE),
    "pk_individual": dict(preprocess=preprocess_pk_individual_table, rating_cols=PK_INDIVIDUAL_RATING_COLUMNS,
                          anchor_cols=PK_INDIVIDUAL_ANCHOR_COLUMNS, columns_type=PK_INDIVIDUAL_COLUMNS_TYPE),
}


def scoring_for(schema: str) -> dict:
    """The strict-F1 scoring config of `schema`: the two explicit entries above,
    else the benchmark's own BenchmarkConfig (benchmark/configs.py) for the
    pipeline - the same preprocess / rating / anchor columns the pk-pe
    benchmark scores with."""
    if schema in SCORING:
        return SCORING[schema]
    cfg = get_benchmark_config(BenchmarkType(schema.replace("_", "-")))
    return dict(preprocess=cfg.preprocess, rating_cols=cfg.rating_cols, anchor_cols=cfg.anchor_cols,
                columns_type=cfg.columns_type)


def make_scorer(schema: str):
    cfg = scoring_for(schema)
    evaluator = TablesF1Evaluator(rating_cols=cfg["rating_cols"], anchor_cols=cfg["anchor_cols"],
                                  columns_type=cfg["columns_type"])

    def strict_f1(pred_csv: str, gold_csv: str) -> float:
        pred = cfg["preprocess"](pred_csv)
        gold = cfg["preprocess"](gold_csv)
        return evaluator.evaluate(gold, pred)["strict"].f1
    return strict_f1


def score_target(dataset: dict, pmid: str) -> tuple[str | None, str]:
    """(csv to score against, label): the paper's gold table when the dataset
    has one, else its reference set (another model's run, e.g. the pk-pe
    benchmark's gpt-4o files - scored the same way but NOT a gold standard,
    hence the label), else (None, "")."""
    gold_dir = dataset.get("gold_dir")
    if gold_dir:
        path = os.path.join(gold_dir, f"{pmid}_baseline.csv")
        if os.path.exists(path):
            return path, "gold"
    ref = dataset.get("reference")
    if ref:
        path = os.path.join(ref["dir"], f"{pmid}{ref['suffix']}")
        if os.path.exists(path):
            return path, ref.get("label", "ref")
    return None, ""


def _paper_f1_from_tables(tables, pick, schema, gold_csv, strict_f1) -> float:
    """Strict F1 of the paper rebuilt from `pick(table_record) -> csv path` for
    each selected table, in source order (what curate_orchestrator's combine
    step does for the winners)."""
    import tempfile
    import pandas as pd

    columns = co.arms._validity.schema_columns(schema)
    frames = [pd.read_csv(pick(t)) for t in tables]
    combined = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    combined = combined.reindex(columns=columns)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        tmp = fh.name
    try:
        combined.to_csv(tmp, index=False)
        return strict_f1(tmp, gold_csv)
    finally:
        os.unlink(tmp)


def arm_f1_from_run(sch: dict, schema: str, gold_csv: str, strict_f1) -> tuple[float, float | None]:
    """(F1_A, F1_B) of one paper from this run's own per-table outputs (see
    the module docstring). F1_B is None when no table escalated."""
    tables = sch["tables"]
    f1_a = _paper_f1_from_tables(tables, lambda t: t["csv_a"], schema, gold_csv, strict_f1)
    if not any(t["escalated"] for t in tables):
        return f1_a, None
    f1_b = _paper_f1_from_tables(tables, lambda t: t["csv_b"] if t["escalated"] else t["csv_a"],
                                 schema, gold_csv, strict_f1)
    return f1_a, f1_b


def select_only(pmids, dataset, schema, pmid_db, work_dir):
    """Run just the schema's table selector on each paper (1 LLM call each),
    print the 1-based selected table indexes with the selector's reasoning,
    and write `selection_<pmid>.json` next to the cascade work dir. Used to
    check a selector change (e.g. the pk_individual selector added after the
    Tier 4 audit) before paying for a full cascade re-run."""
    from extractor.agents.agent_factory import get_agent_llm
    from extractor.agents.agent_utils import extract_pmid_info_to_db

    llm = get_agent_llm()
    selector = co._SELECTORS.get(schema, co.select_pk_summary_tables)
    print(f"selector: {selector.__name__}")
    print(f"{'pmid':<10} {'n_tables':>8} {'selected (1-based)':>20}  reasoning")
    for pmid in pmids:
        html = read_source(pmid, dataset["html_dir"], dataset.get("source_ext"))
        _, _, _, _, tables, _ = extract_pmid_info_to_db(pmid, pmid_db, html_content=html)
        tables = co._as_dataframes(tables or [])
        t0 = time.time()
        selected, raw_indexes, reasoning, token_usage = selector(tables, llm)
        positions = co._positions_of(tables, selected or [])
        record = {
            "pmid": pmid, "schema": schema, "selector": selector.__name__,
            "n_tables": len(tables), "selected_table_indexes": [p + 1 for p in positions],
            "raw_indexes": raw_indexes, "captions": [(t.get("caption") or "")[:120] for t in tables],
            "reasoning": reasoning, "token_usage": token_usage, "seconds": time.time() - t0,
        }
        out_dir = os.path.join(work_dir, "cascade", pmid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"selection_{schema}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, default=str)
        sel = ",".join(str(i) for i in record["selected_table_indexes"]) or "-"
        print(f"{pmid:<10} {len(tables):>8} {sel:>20}  {reasoning}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="pk_summary_qwen36")
    ap.add_argument("--pmid", default=None, help="run just this PMID (one-paper-per-Slurm-task)")
    ap.add_argument("--work-dir", default=None,
                    help="default: .curate_agent_calibration/<dataset> (shares the pmid_info.db cache)")
    ap.add_argument("--threshold", type=float, default=co.vc.DEFAULT_THRESHOLD)
    ap.add_argument("--cheap-model", default=None, help="Ollama tag for Arm A (default: dataset's cheap_model)")
    ap.add_argument("--reference-f1", action="store_true",
                    help="also print/store F1_A_ref / F1_B_ref from the dataset's stored result_f1.log "
                         "collections (off by default: every F1 column then comes from this run)")
    ap.add_argument("--arm-b-f1-log", default=None,
                    help="with --reference-f1: result_f1.log for F1_B_ref (default: the dataset's arm_b_f1_log)")
    ap.add_argument("--arm-b-mode", choices=co.arms.ARM_B_MODES, default=co.arms.DEFAULT_ARM_B_MODE,
                    help="Arm B: 'pipeline' = workflow execution only; 'multiagent' = execution + the "
                         "production per-table verify/correct loop (see curate_agent/arms.py)")
    ap.add_argument("--select-only", action="store_true",
                    help="only run the schema's table selector (1 LLM call per paper) and print the "
                         "selected table indexes + the selector's reasoning; no Arm A/B, no gate")
    args = ap.parse_args(argv)

    dataset = DATASETS[args.dataset]
    schema = dataset["schema"]
    work_dir = args.work_dir or os.path.join(REPO_ROOT, ".curate_agent_calibration", args.dataset)
    os.makedirs(work_dir, exist_ok=True)
    pmid_db = PMIDDB(Path(work_dir) / "pmid_info.db")

    pmids = dataset["pmids"]
    if args.pmid is not None:
        if args.pmid not in pmids:
            raise SystemExit(f"--pmid {args.pmid} is not in dataset {args.dataset!r}: {pmids}")
        pmids = [args.pmid]
    f1_a_ref = f1_b_ref = {}
    if args.reference_f1:
        if not dataset.get("arm_a_f1_log") or not (args.arm_b_f1_log or dataset.get("arm_b_f1_log")):
            raise SystemExit(f"--reference-f1: dataset {args.dataset!r} has no stored reference F1 logs")
        f1_a_ref = load_strict_f1(dataset["arm_a_f1_log"], pmids)
        f1_b_ref = load_strict_f1(args.arm_b_f1_log or dataset["arm_b_f1_log"], pmids)
    strict_f1 = make_scorer(schema)
    cheap_model = args.cheap_model or dataset["cheap_model"]

    print(f"Dataset {args.dataset} (schema={schema}, {len(pmids)} PMIDs), Arm A model {cheap_model}, "
          f"Arm B mode {args.arm_b_mode}, PIPELINE_LLM={os.environ.get('PIPELINE_LLM')} / "
          f"AGENT_LLM={os.environ.get('AGENT_LLM')}, threshold {args.threshold}\n")

    if args.select_only:
        select_only(pmids, dataset, schema, pmid_db, work_dir)
        return
    ref_hdr = f" {'F1_A_ref':>8} {'F1_B_ref':>8}" if args.reference_f1 else ""
    against = "gold" if dataset.get("gold_dir") else (dataset.get("reference") or {}).get("label", "none")
    if against != "gold":
        print(f"NOTE: no gold tables for this dataset - F1 columns are against the '{against}' reference "
              f"set (another model's run), not a gold standard.\n")
    print(f"{'pmid':<10} {'units':>6} {'escalated':>10} {'F1_cascade':>10} {'F1_A':>6} {'F1_B':>6}"
          f"{ref_hdr} {'seconds':>8} {'llm_calls':>9}   vs")

    summary = []
    for pmid in pmids:
        html = read_source(pmid, dataset["html_dir"], dataset.get("source_ext"))
        t0 = time.time()
        result = co.curate(
            pmid, schemas=[schema], html_content=html, pmid_db=pmid_db,
            work_dir=os.path.join(work_dir, "cascade", pmid),
            threshold=args.threshold, cheap_model=cheap_model, arm_b_mode=args.arm_b_mode,
        )
        seconds = time.time() - t0
        sch = result["schemas"][schema]
        gold_csv, scored_against = score_target(dataset, pmid)
        if gold_csv is not None:
            f1_cascade = strict_f1(sch["csv_path"], gold_csv)
            f1_a, f1_b = arm_f1_from_run(sch, schema, gold_csv, strict_f1)
        else:
            f1_cascade = f1_a = f1_b = None
        calls = sum(sch["llm_calls"].values())
        row = {
            "pmid": pmid, "schema": schema, "unit_kind": sch.get("unit_kind"),
            "selected_tables": sch["selected_table_indexes"],
            "escalated_tables": sch["escalated_table_indexes"],
            "units": len(sch["tables"]), "rows": sch["rows"],
            "f1_cascade": f1_cascade, "f1_a": f1_a, "f1_b": f1_b, "scored_against": scored_against,
            "f1_a_ref": f1_a_ref.get(pmid), "f1_b_ref": f1_b_ref.get(pmid),
            "seconds": seconds, "llm_calls": sch["llm_calls"], "llm_calls_total": calls,
            "arm_b_mode": args.arm_b_mode,
            "per_table": [
                {k: t.get(k) for k in ("table_index", "unit_kind", "escalated", "winner", "seconds", "llm_calls")}
                | {"error_rate_a": t["report_a"]["error_rate"],
                   "error_rate_b": (t.get("report_b") or {}).get("error_rate"),
                   "arm_b_final_answer": t.get("arm_b_final_answer")}
                for t in sch["tables"]
            ],
            "csv_path": sch["csv_path"],
        }
        summary.append(row)
        esc = ",".join("text" if i == 0 else str(i) for i in row["escalated_tables"]) or "-"
        fmt = lambda v, w=6: f"{v:>{w}.3f}" if v is not None else f"{'n/a':>{w}}"  # noqa: E731
        ref = f" {fmt(row['f1_a_ref'], 8)} {fmt(row['f1_b_ref'], 8)}" if args.reference_f1 else ""
        print(f"{pmid:<10} {row['units']:>6} {esc:>10} {fmt(f1_cascade, 10)} "
              f"{fmt(row['f1_a'])} {fmt(row['f1_b'])}{ref} {seconds:>8.0f} {calls:>9}   {scored_against or '-'}")

    if len(summary) > 1:
        def avg(k):  # mean over the papers that have the value (a reference log may lack a paper)
            vals = [r[k] for r in summary if r[k] is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        ref = f" {avg('f1_a_ref'):>8.3f} {avg('f1_b_ref'):>8.3f}" if args.reference_f1 else ""
        print(f"\n{'AVERAGE':<10} {'':>6} {'':>10} {avg('f1_cascade'):>10.3f} {avg('f1_a'):>6.3f} "
              f"{avg('f1_b'):>6.3f}{ref} {avg('seconds'):>8.0f} {avg('llm_calls_total'):>9.1f}   {against}")
        n_b = sum(1 for r in summary if r["f1_b"] is not None)
        print(f"(F1_A over all {len(summary)} papers; F1_B over the {n_b} paper(s) where Arm B ran)")

    suffix = f"_{args.pmid}" if args.pmid else ""
    out_path = os.path.join(work_dir, f"cascade_summary{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nSummary -> {out_path}")


if __name__ == "__main__":
    main()
