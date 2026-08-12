#!/usr/bin/env python3
"""Recover curation CSVs that a batch wrote to its scratch dir instead of output.

Why this exists
---------------
Every curation skill ends with

    OUT="${SKILL_OUTPUT_FOLDER:-.}"; mkdir -p "$OUT/<pmid>"

and the orchestrator sets `SKILL_OUTPUT_FOLDER` correctly. But when the model
writes the file with its editor tool rather than running that shell snippet, the
variable never expands, the `:-.` fallback wins, and the CSV lands under the
CWD -- which is the scratch dir. `run_curation()` then finds nothing at the
expected path and reports the skill as failed.

The work was done and paid for in GPU time; only the collection failed. In one
203-paper batch this accounted for 89 of 145 "failed" skills, holding 1,967
curated data rows. The orchestrator now recovers these in-flight (see
`stray_csv_candidates`); this script rescues batches that already ran.

Usage
-----
    # report only -- default, touches nothing
    python scripts/recover_stray_curation_csvs.py --output-root /path/to/output_dir

    # copy the strays into place and rewrite the summary CSVs
    python scripts/recover_stray_curation_csvs.py --output-root /path/... --apply

Layout assumed (what the job scripts produce):
    <output-root>/summary_<job>.csv          one row per paper
    <output-root>/<pmid>/<skill>.csv         where results are supposed to land
    <output-root>/tmp_<job>/scratch/...      where strays are found
"""
import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_summaries(root: Path) -> List[Tuple[Path, List[dict]]]:
    out = []
    for f in sorted(root.glob("summary_*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            out.append((f, list(csv.DictReader(fh))))
    return out


def split(field: Optional[str]) -> List[str]:
    return [x for x in (field or "").split() if x]


def data_rows(path: Path) -> int:
    """Number of non-blank data rows, or -1 if unreadable."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except Exception:
        return -1
    return sum(1 for r in rows[1:] if any((c or "").strip() for c in r))


def find_stray(root: Path, pmid: str, skill: str) -> Optional[Path]:
    """Look where the model actually writes. Most specific candidate first."""
    patterns = [
        f"tmp_*/scratch/{pmid}/{skill}.csv",
        f"tmp_*/scratch/**/{pmid}/{skill}.csv",
    ]
    for pat in patterns:
        hits = sorted(root.glob(pat))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # Several jobs touched this paper; prefer the largest, and say so.
            hits.sort(key=lambda p: p.stat().st_size, reverse=True)
            print(f"    [warn] {pmid}/{skill}: {len(hits)} candidates, taking largest "
                  f"({hits[0]})", file=sys.stderr)
            return hits[0]
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", required=True,
                    help="dir holding summary_*.csv, <pmid>/ results and tmp_*/scratch")
    ap.add_argument("--apply", action="store_true",
                    help="actually copy files and rewrite summaries (default: report only)")
    ap.add_argument("--keep-empty", action="store_true",
                    help="also recover CSVs that have zero data rows (default: report, skip)")
    args = ap.parse_args(argv)

    root = Path(args.output_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    summaries = read_summaries(root)
    if not summaries:
        raise SystemExit(f"no summary_*.csv under {root}")

    missing: List[Tuple[str, str]] = []
    for _f, rows in summaries:
        for r in rows:
            got = set(split(r.get("skills_succeeded")))
            for s in split(r.get("skills_selected")):
                if s not in got:
                    missing.append((r["pmid"], s))

    print(f"output root      : {root}")
    print(f"summary files    : {len(summaries)}")
    print(f"missing (pmid,skill) pairs: {len(missing)}")
    print()

    tally = Counter()
    recovered: Dict[Tuple[str, str], Path] = {}
    for pmid, skill in missing:
        dest = root / pmid / f"{skill}.csv"
        if dest.is_file():
            tally["already in place"] += 1
            continue
        src = find_stray(root, pmid, skill)
        if src is None:
            tally["not found anywhere"] += 1
            continue
        n = data_rows(src)
        if n < 0:
            tally["found but unreadable"] += 1
            continue
        if n == 0 and not args.keep_empty:
            tally["found but empty (skipped)"] += 1
            continue
        tally["recoverable"] += 1
        recovered[(pmid, skill)] = src

    for k, v in tally.most_common():
        print(f"  {v:>4}  {k}")

    total_rows = sum(max(0, data_rows(p)) for p in recovered.values())
    print()
    print(f"recoverable CSVs : {len(recovered)}   holding {total_rows} data rows")

    if not args.apply:
        print()
        print("DRY RUN — nothing written. Re-run with --apply to copy these into place.")
        for (pmid, skill), src in list(recovered.items())[:10]:
            print(f"    {src}  ->  {root / pmid / (skill + '.csv')}")
        if len(recovered) > 10:
            print(f"    ... and {len(recovered) - 10} more")
        return 0

    # ---- apply ----------------------------------------------------------- #
    copied = 0
    for (pmid, skill), src in recovered.items():
        dest = root / pmid / f"{skill}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)          # copy, never move: scratch stays intact
        copied += 1
    print(f"copied {copied} CSV(s) into {root}/<pmid>/")

    # Rewrite the summaries so downstream counts match what is on disk. The
    # originals are kept as .bak -- this rewrites the record of a completed run.
    for f, rows in summaries:
        shutil.copyfile(f, f.with_suffix(".csv.bak"))
        fields = list(rows[0].keys()) if rows else []
        for extra in ("skills_recovered", "skills_failed"):
            if extra not in fields:
                fields.append(extra)
        for r in rows:
            got = split(r.get("skills_succeeded"))
            rec = split(r.get("skills_recovered"))
            changed = False
            for s in split(r.get("skills_selected")):
                if s not in got and (r["pmid"], s) in recovered:
                    got.append(s)
                    rec.append(s)
                    changed = True
            if changed:
                r["skills_succeeded"] = " ".join(got)
                r["skills_recovered"] = " ".join(rec)
                sel = split(r.get("skills_selected"))
                r["status"] = "ok" if len(got) == len(sel) else "partial"
            r.setdefault("skills_recovered", r.get("skills_recovered", ""))
            r.setdefault("skills_failed", r.get("skills_failed", ""))
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
    print(f"rewrote {len(summaries)} summary file(s); originals kept as *.csv.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
