#!/usr/bin/env python3
"""Recover curation CSVs a batch produced but did not deliver to the output dir.

Why this exists
---------------
Every curation skill ends with

    OUT="${SKILL_OUTPUT_FOLDER:-.}"; mkdir -p "$OUT/<pmid>"

and the orchestrator sets `SKILL_OUTPUT_FOLDER` correctly. But when the model
writes the file with its editor tool rather than running that shell snippet, the
variable never expands, the `:-.` fallback wins, and the CSV lands under the CWD
-- the scratch dir. `run_curation()` then finds nothing at the expected path and
reports the skill as failed. The work was done and paid for in GPU time; only
the collection failed.

Two tiers of recovery, in order of confidence:

  STRAY  -- a finished `<skill>.csv` exists, just in the wrong directory.
            Copy it. This is exact: the file is the deliverable.

  STAGE  -- no `<skill>.csv` anywhere, but the skill's own scratch holds its
            numbered stage finals (`13_final.csv`, `05_final.csv`, ...), either
            already merged as `combined_final.csv` or one per curated table.
            Assemble them. This is a RECONSTRUCTION: the skill's real final step
            may do more than concatenate, so these are reported separately and
            recorded as `recovered:stage`.

Safety rules that are not optional
----------------------------------
* Parts are concatenated only when every part's header is byte-identical.
  Mismatched schemas are refused, never glued.
* When the output dir already holds natively-delivered CSVs for a skill, the
  assembled header must match that skill's delivered header. A skill with no
  delivered example anywhere is flagged `unverified-schema` rather than trusted.
* Files are copied, never moved: the scratch tree stays intact.

Usage
-----
    # report only -- the default, touches nothing
    python scripts/recover_stray_curation_csvs.py --output-root /path/to/output_dir

    # write the recovered CSVs and update the summaries
    python scripts/recover_stray_curation_csvs.py --output-root /path/... --apply

Layout assumed (what the job scripts produce):
    <output-root>/summary_<job>.csv                  one row per paper
    <output-root>/<pmid>/<skill>.csv                 where results should land
    <output-root>/tmp_<job>/scratch/                 strays land here
    <output-root>/tmp_<job>/scratch/<skill-scratch>/<pmid>/[<table>/]<NN>_final.csv
"""
import argparse
import csv
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# skill -> (its private scratch dir, the stage file that holds its final rows).
# Read off each ollama_skills/<skill>/SKILL.md; the numbering differs per
# pipeline, so a wildcard here would silently pick up another skill's output.
SKILL_STAGE: Dict[str, Tuple[str, str]] = {
    "pk-summary-curation":      (".pk_curation_scratch",              "13_final.csv"),
    "pk-individual-curation":   (".pk_individual_scratch",            "13_final.csv"),
    "pk-drug-summary":          (".pk_drug_summary_scratch",          "04_final.csv"),
    "pk-drug-individual":       (".pk_drug_individual_scratch",       "04_final.csv"),
    "pk-specimen-summary":      (".pk_specimen_summary_scratch",      "05_final.csv"),
    "pk-specimen-individual":   (".pk_specimen_individual_scratch",   "05_final.csv"),
    "pk-population-summary":    (".pk_population_summary_scratch",    "04_final.csv"),
    "pk-population-individual": (".pk_population_individual_scratch", "05_final.csv"),
    "pe-study-info":            (".pe_study_info_scratch",            "03_final.csv"),
    "pe-study-outcome":         (".pe_study_outcome_scratch",         "05_final.csv"),
}


class Mismatch(Exception):
    """Parts disagree on their header; refuse rather than concatenate."""


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def read_csv(path: Path) -> Tuple[Optional[Tuple[str, ...]], List[List[str]]]:
    """Return (header, non-blank data rows). Header is None if unreadable/empty."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except Exception:
        return None, []
    if not rows:
        return None, []
    body = [r for r in rows[1:] if any((c or "").strip() for c in r)]
    return tuple(rows[0]), body


def split_field(field: Optional[str]) -> List[str]:
    return [x for x in (field or "").split() if x]


def natural_key(p: Path):
    """Sort table_2 before table_10, and before table_3 comes table_2."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def read_summaries(root: Path) -> List[Tuple[Path, List[dict]]]:
    out = []
    for f in sorted(root.glob("summary_*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            out.append((f, list(csv.DictReader(fh))))
    return out


def delivered_headers(root: Path) -> Dict[str, Tuple[str, ...]]:
    """The header each skill uses when it delivers natively -- our ground truth.

    Taken by majority over every <pmid>/<skill>.csv already in place, so one
    malformed delivery cannot redefine what the schema is.
    """
    seen: Dict[str, Counter] = defaultdict(Counter)
    for skill in SKILL_STAGE:
        for f in root.glob(f"*/{skill}.csv"):
            head, _ = read_csv(f)
            if head:
                seen[skill][head] += 1
    return {s: c.most_common(1)[0][0] for s, c in seen.items() if c}


def find_stray(root: Path, pmid: str, skill: str) -> Optional[Path]:
    """A finished <skill>.csv sitting in the wrong directory.

    Every pattern here MUST contain the pmid. A scratch dir is shared by all ~10
    papers of its job, so `tmp_N/scratch/<skill>.csv` -- no pmid in the path --
    cannot be attributed: it belongs to whichever paper wrote it last, and every
    other paper in that job would claim it too. Those are counted separately as
    unattributable and left alone.
    """
    for pat in (f"tmp_*/scratch/{pmid}/{skill}.csv",
                f"tmp_*/scratch/**/{pmid}/{skill}.csv"):
        hits = sorted(root.glob(pat))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # Several jobs curated this paper; same pmid, so attribution is safe.
            hits.sort(key=lambda p: p.stat().st_size, reverse=True)
            print(f"    [warn] {pmid}/{skill}: {len(hits)} stray candidates for this pmid, "
                  f"taking largest", file=sys.stderr)
            return hits[0]
    return None


def has_unattributable(root: Path, skill: str) -> bool:
    """True when a pmid-less <skill>.csv exists in some job's scratch root."""
    return any(root.glob(f"tmp_*/scratch/{skill}.csv"))


def find_stage_parts(root: Path, pmid: str, skill: str) -> Tuple[str, List[Path]]:
    """Locate the skill's own stage finals. Returns (kind, parts)."""
    scratch_dir, final_name = SKILL_STAGE[skill]
    combined = sorted(root.glob(f"tmp_*/scratch/{scratch_dir}/{pmid}/combined_final.csv"))
    if combined:
        # The skill already merged its tables; prefer that over re-merging.
        return "combined", combined[:1]
    per_table = sorted(root.glob(f"tmp_*/scratch/{scratch_dir}/{pmid}/*/{final_name}"),
                       key=natural_key)
    if per_table:
        return "parts", per_table
    single = sorted(root.glob(f"tmp_*/scratch/{scratch_dir}/{pmid}/{final_name}"))
    if single:
        return "single", single[:1]
    return "none", []


def assemble(parts: Sequence[Path]) -> Tuple[Tuple[str, ...], List[List[str]]]:
    """Concatenate parts, refusing any header disagreement."""
    header: Optional[Tuple[str, ...]] = None
    body: List[List[str]] = []
    for p in parts:
        h, rows = read_csv(p)
        if h is None:
            continue
        if header is None:
            header = h
        elif h != header:
            raise Mismatch(f"{p.name}: {list(h)[:3]}... != {list(header)[:3]}...")
        body.extend(rows)
    if header is None:
        raise Mismatch("no readable part")
    return header, body


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", required=True,
                    help="dir holding summary_*.csv, <pmid>/ results and tmp_*/scratch")
    ap.add_argument("--apply", action="store_true",
                    help="write recovered CSVs and update summaries (default: report only)")
    ap.add_argument("--keep-empty", action="store_true",
                    help="also recover results with zero data rows (default: skip)")
    ap.add_argument("--no-stage", action="store_true",
                    help="only recover finished <skill>.csv strays; skip stage reconstruction")
    ap.add_argument("--allow-unverified-schema", action="store_true",
                    help="accept an assembled header for a skill that has no delivered example")
    args = ap.parse_args(argv)

    root = Path(args.output_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    summaries = read_summaries(root)
    if not summaries:
        raise SystemExit(f"no summary_*.csv under {root}")

    known = delivered_headers(root)
    missing: List[Tuple[str, str]] = []
    for _f, rows in summaries:
        for r in rows:
            got = set(split_field(r.get("skills_succeeded")))
            for s in split_field(r.get("skills_selected")):
                if s not in got:
                    missing.append((r["pmid"], s))

    print(f"output root       : {root}")
    print(f"summary files     : {len(summaries)}")
    print(f"missing (pmid,skill): {len(missing)}")
    print(f"skills with a known delivered header: {len(known)}/{len(SKILL_STAGE)}")
    print()

    tally = Counter()
    # (pmid, skill) -> (kind, header, rows, sources)
    plan: Dict[Tuple[str, str], Tuple[str, Tuple[str, ...], List[List[str]], List[Path]]] = {}

    for pmid, skill in missing:
        if (root / pmid / f"{skill}.csv").is_file():
            tally["already in place"] += 1
            continue
        if skill not in SKILL_STAGE:
            tally["unknown skill"] += 1
            continue

        stray = find_stray(root, pmid, skill)
        if stray is not None:
            head, rows = read_csv(stray)
            if head is None:
                tally["stray unreadable"] += 1
                continue
            if not rows and not args.keep_empty:
                tally["stray, zero rows (skipped)"] += 1
                continue
            tally["STRAY recoverable"] += 1
            plan[(pmid, skill)] = ("stray", head, rows, [stray])
            continue

        if args.no_stage:
            tally["no stray (stage disabled)"] += 1
            continue

        kind, parts = find_stage_parts(root, pmid, skill)
        if kind == "none":
            if has_unattributable(root, skill):
                tally["nothing attributable (pmid-less file exists)"] += 1
            else:
                tally["nothing on disk"] += 1
            continue
        try:
            head, rows = assemble(parts)
        except Mismatch as e:
            tally["REFUSED: header mismatch"] += 1
            print(f"    [refuse] {pmid}/{skill}: {e}", file=sys.stderr)
            continue
        expected = known.get(skill)
        if expected is None and not args.allow_unverified_schema:
            tally["REFUSED: schema unverified"] += 1
            continue
        if expected is not None and head != expected:
            tally["REFUSED: header != delivered"] += 1
            print(f"    [refuse] {pmid}/{skill}: assembled header differs from delivered",
                  file=sys.stderr)
            continue
        if not rows and not args.keep_empty:
            tally["stage, zero rows (skipped)"] += 1
            continue
        tally[f"STAGE recoverable ({kind})"] += 1
        plan[(pmid, skill)] = ("stage", head, rows, list(parts))

    for k, v in tally.most_common():
        print(f"  {v:>4}  {k}")

    n_stray = sum(1 for v in plan.values() if v[0] == "stray")
    n_stage = len(plan) - n_stray
    rows_stray = sum(len(v[2]) for v in plan.values() if v[0] == "stray")
    rows_stage = sum(len(v[2]) for v in plan.values() if v[0] == "stage")
    print()
    print(f"  stray (exact deliverable) : {n_stray:>4} skills, {rows_stray:>6} rows")
    print(f"  stage (reconstructed)     : {n_stage:>4} skills, {rows_stage:>6} rows")
    print(f"  TOTAL                     : {len(plan):>4} skills, {rows_stray + rows_stage:>6} rows")

    if not args.apply:
        print()
        print("DRY RUN — nothing written. Re-run with --apply.")
        for (pmid, skill), (kind, _h, rows, src) in list(plan.items())[:8]:
            print(f"    [{kind}] {pmid}/{skill}.csv  <- {len(src)} file(s), {len(rows)} rows")
        if len(plan) > 8:
            print(f"    ... and {len(plan) - 8} more")
        return 0

    # ---- apply ------------------------------------------------------------ #
    manifest_path = root / "recovery_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        mw = csv.writer(fh)
        mw.writerow(["pmid", "skill", "kind", "data_rows", "n_sources", "sources"])
        for (pmid, skill), (kind, head, rows, src) in sorted(plan.items()):
            dest = root / pmid / f"{skill}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if kind == "stray":
                shutil.copyfile(src[0], dest)       # exact file; do not rewrite it
            else:
                # lineterminator="\n": csv.writer defaults to \r\n, which would
                # make every reconstructed file differ from a natively delivered
                # one by its line endings alone.
                with open(dest, "w", newline="", encoding="utf-8") as out:
                    w = csv.writer(out, lineterminator="\n")
                    w.writerow(head)
                    w.writerows(rows)
            mw.writerow([pmid, skill, kind, len(rows), len(src),
                         " | ".join(str(s) for s in src)])
    print()
    print(f"wrote {len(plan)} CSV(s); provenance -> {manifest_path}")

    for f, rows in summaries:
        shutil.copyfile(f, f.with_suffix(".csv.bak"))
        fields = list(rows[0].keys()) if rows else []
        for extra in ("skills_recovered", "skills_failed"):
            if extra not in fields:
                fields.append(extra)
        for r in rows:
            got = split_field(r.get("skills_succeeded"))
            rec = split_field(r.get("skills_recovered"))
            changed = False
            for s in split_field(r.get("skills_selected")):
                key = (r["pmid"], s)
                if s not in got and key in plan:
                    got.append(s)
                    # keep the kind: a reconstruction is not a delivered result
                    rec.append(s if plan[key][0] == "stray" else f"{s}:stage")
                    changed = True
            if changed:
                r["skills_succeeded"] = " ".join(got)
                r["skills_recovered"] = " ".join(rec)
                r["status"] = ("ok" if len(got) == len(split_field(r.get("skills_selected")))
                               else "partial")
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
    print(f"rewrote {len(summaries)} summary file(s); originals kept as *.csv.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
