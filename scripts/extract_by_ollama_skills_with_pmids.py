#!/usr/bin/env python3
"""Batch-curate PK/PE papers by chaining the ./ollama_skills together.

The ollama_skills are *agentic* prompt-skills (they read SKILL.md, run scripts,
call the model, loop), so they need an agent runtime. This script is the
orchestrator that small open models cannot be: for each paper it runs the chain
deterministically in Python, invoking the Claude Code CLI (`claude -p`) once per
skill. Use it with Claude Code pointed at your Ollama server (set the backend the
usual way, or pass --model).

Per paper (one row of the manifest CSV):

  1. PREPARE  (deterministic)  pk-pe-prepare/scripts/prepare_paper.py on the HTML
                               -> $SCRATCH/.paper_assets/<pmid>/
  2. ROUTE    (model)          `claude -p "use pk-pe-route ..."`
                               -> reads $SCRATCH/.pk_pe_route_scratch/<pmid>/selected_pipelines.json
                               (skipped when --pipelines forces a fixed list)
  3. CURATE   (model, looped)  for each selected skill: `claude -p "use <skill> ..."`
                               -> $OUTPUT/<pmid>/<skill>.csv

The two env vars the skills honor are set per run: SKILL_SCRATCH_FOLDER (working
files) and SKILL_OUTPUT_FOLDER (result CSVs). Progress is written incrementally to
a summary CSV so a crash does not lose completed work; --resume skips papers/skills
whose result CSV already exists.

Usage:
    python scripts/extract_by_ollama_skills_with_pmids.py -f manifest.csv -o ./out
    python scripts/extract_by_ollama_skills_with_pmids.py -f manifest.csv -o ./out \
        -p pk_summary pk_individual          # force pipelines, skip routing
    python scripts/extract_by_ollama_skills_with_pmids.py -f manifest.csv -o ./out --dry-run

Manifest CSV columns (header row required): `pmid, html_path`.
"""
import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Pipeline label (PipelineTypeEnum value) -> ollama skill folder name. Mirrors the
# generated ollama_skills/pk-pe-route/scripts/pipeline_skill_map.py; kept here so a
# --pipelines override can accept either labels or skill names.
LABEL_TO_SKILL = {
    "pk_summary": "pk-summary-curation",
    "pk_individual": "pk-individual-curation",
    "pk_specimen_summary": "pk-specimen-summary",
    "pk_specimen_individual": "pk-specimen-individual",
    "pk_drug_summary": "pk-drug-summary",
    "pk_drug_individual": "pk-drug-individual",
    "pk_population_summary": "pk-population-summary",
    "pk_population_individual": "pk-population-individual",
    "pe_study_info": "pe-study-info",
    "pe_study_outcome": "pe-study-outcome",
}
ALL_SKILLS = set(LABEL_TO_SKILL.values())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("extract_by_ollama_skills")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-f", "--manifest", required=True,
                    help="CSV with columns: pmid, html_path")
    ap.add_argument("-o", "--output", required=True,
                    help="output dir for result CSVs (becomes $SKILL_OUTPUT_FOLDER); "
                         "each result is <output>/<pmid>/<skill>.csv")
    ap.add_argument("-s", "--scratch", default=None,
                    help="working dir for intermediates (becomes $SKILL_SCRATCH_FOLDER); "
                         "default: <output>/.scratch")
    ap.add_argument("-p", "--pipelines", nargs="*", default=None,
                    help="force these pipelines for every paper, skipping routing. "
                         "Accepts labels (pk_summary) or skill names (pk-summary-curation).")
    ap.add_argument("--skills-dir", default=None,
                    help="path to the ollama_skills bundle (default: ./ollama_skills "
                         "next to this repo)")
    ap.add_argument("--claude-bin", default="claude", help="Claude Code CLI (default: claude)")
    ap.add_argument("--model", default=None, help="model passed to `claude --model`")
    ap.add_argument("--permission-mode", default=None,
                    help="value for `claude --permission-mode`; if omitted, the script "
                         "uses --dangerously-skip-permissions for unattended runs")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-`claude` call timeout in seconds (default: 1800)")
    ap.add_argument("--resume", action="store_true",
                    help="skip a (pmid, skill) whose result CSV already exists")
    ap.add_argument("-j", "--job-id", default=None,
                    help="summary filename suffix (default: process id)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned steps; do not run prepare or claude")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def read_manifest(path: Path) -> List[Tuple[str, str]]:
    """Return [(pmid, html_path), ...]. Header row required; tolerant of column aliases."""
    rows: List[Tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit(f"manifest {path} is empty or has no header")
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        pmid_col = cols.get("pmid")
        html_col = cols.get("html_path") or cols.get("html_file_path") or cols.get("html")
        if not pmid_col or not html_col:
            raise SystemExit(
                f"manifest must have 'pmid' and 'html_path' columns; got {reader.fieldnames}"
            )
        for r in reader:
            pmid = (r.get(pmid_col) or "").strip()
            html = (r.get(html_col) or "").strip()
            if pmid and html:
                rows.append((pmid, html))
    if not rows:
        raise SystemExit(f"no usable rows in manifest {path}")
    return rows


def resolve_pipelines(tokens: List[str]) -> List[str]:
    """Map a --pipelines list (labels or skill names) to skill folder names."""
    out: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok in LABEL_TO_SKILL:
            skill = LABEL_TO_SKILL[tok]
        elif tok in ALL_SKILLS:
            skill = tok
        else:
            raise SystemExit(
                f"unknown pipeline {tok!r}; valid labels: {sorted(LABEL_TO_SKILL)} "
                f"or skill names: {sorted(ALL_SKILLS)}"
            )
        if skill not in out:
            out.append(skill)
    return out


# --------------------------------------------------------------------------- #
# Run-dir setup: a working dir with .claude/skills/ so `claude` discovers them
# --------------------------------------------------------------------------- #
def setup_workdir(scratch: Path, skills_dir: Path) -> Path:
    """scratch becomes the claude working dir; symlink each skill into .claude/skills/."""
    scratch.mkdir(parents=True, exist_ok=True)
    skills_install = scratch / ".claude" / "skills"
    skills_install.mkdir(parents=True, exist_ok=True)
    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        link = skills_install / child.name
        if link.is_symlink() or link.exists():
            continue
        os.symlink(child.resolve(), link)
    return scratch


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def run_prepare(skills_dir: Path, html_path: Path, pmid: str, scratch: Path,
                dry_run: bool) -> Path:
    """Stage the HTML as <pmid>.<ext> then run prepare_paper.py -> scratch/.paper_assets/<pmid>/."""
    prep = skills_dir / "pk-pe-prepare" / "scripts" / "prepare_paper.py"
    assets_root = scratch / ".paper_assets"
    staged = scratch / ".inputs" / f"{pmid}{html_path.suffix or '.html'}"
    target = assets_root / pmid
    if dry_run:
        logger.info("[dry-run] prepare %s -> %s", html_path, target)
        return target
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(html_path, staged)
    cmd = [sys.executable, str(prep), str(staged), "--out", str(assets_root)]
    logger.info("prepare: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not target.is_dir():
        raise RuntimeError(f"prepare produced no {target} (pmid/base-name mismatch?)")
    return target


def build_claude_cmd(args: argparse.Namespace, prompt: str,
                     extra_dirs: List[Path]) -> List[str]:
    cmd = [args.claude_bin, "-p", prompt]
    for d in extra_dirs:
        cmd += ["--add-dir", str(d)]
    if args.model:
        cmd += ["--model", args.model]
    if args.permission_mode:
        cmd += ["--permission-mode", args.permission_mode]
    else:
        cmd += ["--dangerously-skip-permissions"]
    return cmd


def run_claude(args: argparse.Namespace, prompt: str, cwd: Path, env: Dict[str, str],
               extra_dirs: List[Path], log_path: Path) -> str:
    """Run one `claude -p` call. Returns a status STRING, not a bool.

    The caller needs to tell a timeout apart from a non-zero exit apart from a
    clean run that simply produced no file: those are three different problems
    and only the timeout is worth re-running as-is. Collapsing them to a bool is
    what made a whole batch report "partial" with no recoverable reason.
    """
    cmd = build_claude_cmd(args, prompt, extra_dirs)
    if args.dry_run:
        logger.info("[dry-run] claude (cwd=%s): %s", cwd, prompt.splitlines()[0])
        return "dry-run"
    logger.info("claude: %s", prompt.splitlines()[0])
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), env=env, timeout=args.timeout,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        logger.error("claude timed out after %ss", args.timeout)
        log_path.write_text(f"TIMEOUT after {args.timeout}s\nPROMPT:\n{prompt}\n", encoding="utf-8")
        return "timeout"
    log_path.write_text(
        f"PROMPT:\n{prompt}\n\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        logger.error("claude exited %s (see %s)", proc.returncode, log_path)
        return f"exit:{proc.returncode}"
    return "ok"


def run_route(args: argparse.Namespace, pmid: str, scratch: Path, env: Dict[str, str],
              extra_dirs: List[Path], logdir: Path) -> Tuple[Optional[str], List[str]]:
    """Run pk-pe-route; return (paper_type, [skill names])."""
    prompt = (
        f"Use the pk-pe-route skill to decide which curation pipelines apply to paper {pmid}. "
        f"Its prepared assets are already in .paper_assets/{pmid}/ (paper_text.md, abstract.md, "
        f"table_*.md / table_*.html, manifest.json). Run the skill's identify and design stages "
        f"and its deterministic dispatch map, writing selected_pipelines.json. Do NOT curate any data."
    )
    status = run_claude(args, prompt, scratch, env, extra_dirs, logdir / f"{pmid}_route.log")
    sel_path = scratch / ".pk_pe_route_scratch" / pmid / "selected_pipelines.json"
    if args.dry_run:
        logger.info("[dry-run] would read %s", sel_path)
        return "DRYRUN", []
    if status != "ok" or not sel_path.is_file():
        logger.error("route produced no %s", sel_path)
        return None, []
    data = json.loads(sel_path.read_text(encoding="utf-8"))
    skills = [e["skill"] for e in data.get("selected", []) if e.get("skill") in ALL_SKILLS]
    return data.get("paper_type"), skills


def stray_csv_candidates(scratch: Path, output: Path, pmid: str, skill: str) -> List[Path]:
    """Places a skill has been observed to write its CSV other than the real one.

    The skills end with `OUT="${SKILL_OUTPUT_FOLDER:-.}"; mkdir -p "$OUT/<pmid>"`.
    When the model writes the file with its editor tool instead of running that
    snippet, the variable never expands, the `:-.` fallback wins, and the CSV
    lands under the CWD -- which is the scratch dir. The run looks like a
    failure while the curated rows sit on disk one directory away.
    """
    return [
        scratch / pmid / f"{skill}.csv",   # ./<pmid>/<skill>.csv  -- by far the most common
        scratch / f"{skill}.csv",          # ./<skill>.csv
        output / f"{skill}.csv",           # output/<skill>.csv, pmid level dropped
    ]


def run_curation(args: argparse.Namespace, skill: str, pmid: str, scratch: Path,
                 output: Path, env: Dict[str, str], extra_dirs: List[Path],
                 logdir: Path) -> Tuple[Optional[Path], str]:
    """Run one curation skill; return (result CSV path or None, reason).

    `reason` is recorded per skill in the summary CSV so a batch that comes back
    incomplete says WHY -- ok / resumed / recovered / timeout / exit:N / absent.
    """
    result = output / pmid / f"{skill}.csv"
    if args.resume and result.is_file():
        logger.info("resume: %s already exists, skipping", result)
        return result, "resumed"
    # Pin the ABSOLUTE destination. Naming only the basename left the directory
    # to a shell variable the model does not always expand.
    prompt = (
        f"Use the {skill} skill to curate paper {pmid}. Its prepared assets are in "
        f".paper_assets/{pmid}/ (use paper_text.md and abstract.md for full-text skills, or the "
        f"relevant table_*.html for table skills). Follow the skill's ENTIRE procedure end to end "
        f"-- every stage, the verify/correct step, and the final 'Write out the result' step.\n\n"
        f"Write the result to exactly this absolute path:\n    {result}\n"
        f"Create the parent directory if it does not exist. Do not write it anywhere else, do not "
        f"write it relative to the current directory, and do not rely on $SKILL_OUTPUT_FOLDER being "
        f"expanded. Do not stop early or skip stages."
    )
    status = run_claude(args, prompt, scratch, env, extra_dirs, logdir / f"{pmid}_{skill}.log")
    if args.dry_run:
        return result, "dry-run"
    if result.is_file():
        return result, "ok"

    # Not at the expected path. Before calling it a failure, look where the
    # model actually writes -- this accounted for 61% of one batch's "failures".
    for cand in stray_csv_candidates(scratch, output, pmid, skill):
        if cand.is_file() and cand.resolve() != result.resolve():
            result.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cand, result)
            logger.warning("recovered %s for %s from %s", skill, pmid, cand)
            return result, "recovered"

    reason = status if status != "ok" else "absent"
    logger.error("curation %s for %s produced no CSV (%s)", skill, pmid, reason)
    return None, reason


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    skills_dir = Path(args.skills_dir).resolve() if args.skills_dir else repo / "ollama_skills"
    if not skills_dir.is_dir():
        raise SystemExit(f"skills dir not found: {skills_dir}")

    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    scratch = Path(args.scratch).resolve() if args.scratch else output / ".scratch"
    output.mkdir(parents=True, exist_ok=True)
    logdir = scratch / ".logs"
    logdir.mkdir(parents=True, exist_ok=True)

    forced = resolve_pipelines(args.pipelines) if args.pipelines else None
    rows = read_manifest(manifest)
    setup_workdir(scratch, skills_dir)

    env = dict(os.environ)
    env["SKILL_SCRATCH_FOLDER"] = str(scratch)
    env["SKILL_OUTPUT_FOLDER"] = str(output)
    extra_dirs = [scratch, output, skills_dir]

    job = args.job_id or str(os.getpid())
    summary_path = output / f"summary_{job}.csv"
    # skills_failed carries `skill:reason` per skill that produced nothing, and
    # skills_recovered names those whose CSV had to be rescued from the scratch
    # dir. Without them an incomplete batch reports only THAT it is incomplete.
    summary_fields = ["pmid", "html_path", "paper_type", "skills_selected",
                      "skills_succeeded", "skills_recovered", "skills_failed",
                      "status", "error"]
    summary_fh = open(summary_path, "w", newline="", encoding="utf-8")
    summary = csv.DictWriter(summary_fh, fieldnames=summary_fields)
    summary.writeheader()
    summary_fh.flush()

    logger.info("papers=%d  skills_dir=%s  scratch=%s  output=%s%s",
                len(rows), skills_dir, scratch, output,
                f"  forced={forced}" if forced else "  (routing per paper)")

    for idx, (pmid, html) in enumerate(rows, 1):
        logger.info("=== [%d/%d] pmid=%s ===", idx, len(rows), pmid)
        rec = {"pmid": pmid, "html_path": html, "paper_type": "", "skills_selected": "",
               "skills_succeeded": "", "skills_recovered": "", "skills_failed": "",
               "status": "", "error": ""}
        try:
            html_path = Path(html)
            if not html_path.is_absolute():
                html_path = (manifest.parent / html_path).resolve()
            if not args.dry_run and not html_path.is_file():
                raise FileNotFoundError(f"html not found: {html_path}")

            run_prepare(skills_dir, html_path, pmid, scratch, args.dry_run)

            if forced is not None:
                paper_type, skills = "forced", list(forced)
            else:
                paper_type, skills = run_route(args, pmid, scratch, env, extra_dirs, logdir)
                if paper_type is None:
                    raise RuntimeError("routing failed")

            rec["paper_type"] = paper_type or ""
            rec["skills_selected"] = " ".join(skills)
            if not skills:
                rec["status"] = "no_pipelines"  # e.g. Neither, or empty selection
                logger.info("pmid=%s: no pipelines selected (%s)", pmid, paper_type)
                continue

            succeeded, recovered, failed = [], [], []
            for skill in skills:
                csv_path, reason = run_curation(args, skill, pmid, scratch, output,
                                                env, extra_dirs, logdir)
                if csv_path is not None:
                    succeeded.append(skill)
                    if reason == "recovered":
                        recovered.append(skill)
                else:
                    failed.append(f"{skill}:{reason}")
            rec["skills_succeeded"] = " ".join(succeeded)
            rec["skills_recovered"] = " ".join(recovered)
            rec["skills_failed"] = " ".join(failed)
            rec["status"] = "ok" if len(succeeded) == len(skills) else "partial"
        except Exception as e:  # noqa: BLE001 - per-paper isolation
            rec["status"] = "error"
            rec["error"] = str(e)
            logger.exception("pmid=%s failed", pmid)
        finally:
            summary.writerow(rec)
            summary_fh.flush()

    summary_fh.close()
    logger.info("done. summary -> %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
