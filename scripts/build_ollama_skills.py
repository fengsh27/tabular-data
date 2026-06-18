#!/usr/bin/env python3
"""Generate ./ollama_skills/ from ./skills/pk-pe-curation/.

The nested `skills/pk-pe-curation/` layout (one orchestrator skill containing the
10 pipelines as passive sub-procedures under `pipelines/`, all sharing
`curation-common/`) works well for Claude but causes smaller open models (Qwen)
to skip stages: when a pipeline's procedure is only reachable through orchestrator
indirection it is not "active content", and a weak model collapses the plan.

This script emits an alternative `ollama_skills/` bundle where:

  * each of the 10 curation pipelines is a FIRST-CLASS, independently-triggerable
    skill (its `procedure.md` becomes `SKILL.md` with trigger frontmatter), so its
    full procedure loads as active content;
  * each pipeline is FULLY SELF-CONTAINED -- it carries its own copies of every
    script and resource it needs; there is NO shared `curation-common/`;
  * a standalone `pk-pe-prepare` front-door skill converts a raw paper into the
    canonical input layout, and a standalone `pk-pe-route` selector classifies the
    paper and picks the applicable pipelines. There is NO
    `pk-pe-curation` router skill: full orchestration (prepare + route + dispatch
    in one) is the Claude bundle's job and is too much for small open models, which
    are driven instead by triggering one self-contained skill at a time.

`./skills/` is never modified. Re-run this script to regenerate `ollama_skills/`
after editing `./skills/`.

Usage:
    python scripts/build_ollama_skills.py
"""

import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "skills", "pk-pe-curation")
DST = os.path.join(REPO, "ollama_skills")
CC_SCRIPTS = os.path.join(SRC, "curation-common", "scripts")
CC_ROOT = os.path.join(SRC, "curation-common")
# generator-owned assets that are NOT in skills/ (kept out so skills/ stays the
# unchanged Claude bundle). The XML-capable prepare_paper.py lives here.
ASSETS = os.path.join(REPO, "scripts", "assets")

# --- Per-pipeline manifest -------------------------------------------------
# `extra_scripts` are pulled from curation-common/scripts/ into the skill's own
# scripts/. Scripts already living in the pipeline's own scripts/ (e.g.
# clean_individual_rows.py) are copied verbatim with the pipeline and need no
# entry here. `docs` are resource .md files pulled from curation-common/ into
# the skill root.
PIPELINES = {
    "pk-summary-curation": {
        "desc": "Curate aggregate/summary pharmacokinetics (PK) tables (mean / median / SD / range across a cohort) from a biomedical paper into a normalized 19-column dataset. Use for summary PK tables; for per-subject tables use pk-individual-curation.",
        "extra_scripts": ["verify_provenance.py", "html_to_markdown_table.py", "requirements.txt"],
        "docs": ["verify_and_correct.md"],
    },
    "pk-individual-curation": {
        "desc": "Curate individual-subject pharmacokinetics (PK) tables (one row per subject per parameter) from a biomedical paper into a normalized 12-column dataset. Use when tables report per-individual PK values; do NOT use for summary/aggregate tables (use pk-summary-curation).",
        "extra_scripts": ["verify_provenance.py", "html_to_markdown_table.py", "requirements.txt"],
        "docs": ["verify_and_correct.md"],
    },
    "pe-study-outcome": {
        "desc": "Curate pharmacoepidemiology (PE) study-outcome tables from a paper into a normalized 12-column dataset (effect estimates, confidence intervals, p-values per outcome).",
        "extra_scripts": ["verify_provenance.py", "html_to_markdown_table.py", "clean_pe_outcome_rows.py", "requirements.txt"],
        "docs": ["verify_and_correct.md"],
    },
    "pe-study-info": {
        "desc": "Extract pharmacoepidemiology (PE) study metadata (design, population, exposure, outcome definitions) from a paper's full text into a 10-column single-row dataset.",
        "extra_scripts": ["verify_provenance.py"],
        "docs": ["verify_and_correct.md"],
    },
    "pk-drug-summary": {
        "desc": "Extract summary (cohort-level) drug dosing regimens (dose amount / unit / frequency / route) from a PK paper's full text into an 11-column dataset. For per-patient dosing use pk-drug-individual.",
        "extra_scripts": ["verify_provenance.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
    "pk-drug-individual": {
        "desc": "Extract per-patient drug dosing regimens from a PK paper's full text into an 11-column dataset. For cohort-level dosing use pk-drug-summary.",
        "extra_scripts": ["verify_provenance.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
    "pk-population-summary": {
        "desc": "Extract summary population/demographic characteristics with statistics from a PK paper's full text into a 15-column dataset. For per-patient characteristics use pk-population-individual.",
        "extra_scripts": ["verify_provenance.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
    "pk-population-individual": {
        "desc": "Extract per-patient population/demographic characteristics from a PK paper's full text into a 9-column dataset. For cohort-level stats use pk-population-summary.",
        "extra_scripts": ["verify_provenance.py", "clean_population_individual_rows.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
    "pk-specimen-summary": {
        "desc": "Extract summary (cohort-level) specimen-sampling information (specimen type, sampling times) from a PK paper's full text into a 9-column dataset. For per-patient sampling use pk-specimen-individual.",
        "extra_scripts": ["verify_provenance.py", "clean_specimen_rows.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
    "pk-specimen-individual": {
        "desc": "Extract per-patient specimen-sampling information from a PK paper's full text into a 9-column dataset. For cohort-level sampling use pk-specimen-summary.",
        "extra_scripts": ["verify_provenance.py", "clean_specimen_rows.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
}

PATH_NOTE = (
    "> **Self-contained skill.** Every `scripts/…` path and every resource file "
    "(`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this "
    "skill's own directory**. This skill shares nothing with other skills — when run as "
    "an installed skill, resolve these paths under this skill's folder.\n"
)

# Injected into every generated skill. Lets the user redirect all of a skill's
# working files (prepared inputs + per-run scratch) under one configurable base via
# the SKILL_SCRATCH_FOLDER env var, instead of scattering dot-dirs in the project
# root. Phrased generically (folder-agnostic) so it applies to all 12 skills.
SCRATCH_NOTE = (
    "> **Working-directory base (read this first).** Every `./.…` path this skill "
    "uses below — the prepared inputs in `./.paper_assets/<pmid>/` and any "
    "`./.…_scratch/<pmid>/` intermediates this skill writes — is relative to one "
    "base directory. Resolve it **once, before any file operation**: if the "
    "environment variable `SKILL_SCRATCH_FOLDER` is set (run "
    "`echo \"$SKILL_SCRATCH_FOLDER\"` to check), that is the base — e.g. write to "
    "`\"$SKILL_SCRATCH_FOLDER\"/.<name>_scratch/<pmid>/` and read from "
    "`\"$SKILL_SCRATCH_FOLDER\"/.paper_assets/<pmid>/`. Otherwise the base is the "
    "user's current working directory (use the paths exactly as written below). "
    "Create directories with `mkdir -p` and keep the same base for every read and "
    "write.\n"
)


def rewrite_paths(text: str) -> str:
    """Rewrite shared curation-common/ references to this skill's local paths."""
    text = text.replace("curation-common/scripts/", "scripts/")
    text = text.replace("curation-common/verify_and_correct.md", "verify_and_correct.md")
    text = text.replace("curation-common/refine_population.md", "refine_population.md")
    # any remaining bare backticked dir reference, e.g. `curation-common/`
    text = text.replace("`curation-common/`", "`this skill`")
    text = text.replace("`curation-common`", "`this skill`")
    if "curation-common" in text:
        # surface anything unexpected rather than silently shipping a broken path
        raise RuntimeError("unhandled 'curation-common' reference remains after rewrite")
    return text


def output_note(name: str) -> str:
    """Final section appended to each curation skill: write the deliverable CSV to
    SKILL_OUTPUT_FOLDER. Destination filename is the skill's own name, so multiple
    pipelines on one paper collect as `<pmid>/<skill>.csv`. Plain-concatenated (not
    an f-string) to keep the `${...}` shell expansion literal."""
    return (
        "## Write out the result (do this last)\n"
        "When the procedure above finishes, copy its **final deliverable CSV** (the "
        "`combined_final.csv` / `NN_final.csv` written by the last stage) to the "
        "output location, leaving the scratch copy in place:\n\n"
        "```bash\n"
        '# output base: $SKILL_OUTPUT_FOLDER if set, else the current directory\n'
        'OUT="${SKILL_OUTPUT_FOLDER:-.}"; mkdir -p "$OUT/<pmid>"\n'
        'cp <final-csv-in-scratch> "$OUT/<pmid>/' + name + '.csv"\n'
        "```\n\n"
        "If `SKILL_OUTPUT_FOLDER` is unset this writes `./<pmid>/" + name + ".csv` in "
        "the user's current working directory. Then tell the user the exact path you "
        "wrote.\n"
    )


def make_skill_md(name: str, desc: str, body: str) -> str:
    body = rewrite_paths(body)
    fm = f"---\nname: {name}\ndescription: {desc}\n---\n\n"
    return fm + PATH_NOTE + "\n" + SCRATCH_NOTE + "\n" + body + "\n" + output_note(name)


def copy_pipeline(name: str, spec: dict) -> dict:
    src = os.path.join(SRC, "pipelines", name)
    dst = os.path.join(DST, name)
    shutil.copytree(src, dst)  # brings procedure.md, prompts/, and any own scripts/

    # procedure.md -> SKILL.md (+ frontmatter, path note, rewrites)
    proc = os.path.join(dst, "procedure.md")
    with open(proc, encoding="utf-8") as fh:
        body = fh.read()
    os.remove(proc)
    with open(os.path.join(dst, "SKILL.md"), "w", encoding="utf-8") as fh:
        fh.write(make_skill_md(name, spec["desc"], body))

    # rewrite paths inside every prompt file too
    pdir = os.path.join(dst, "prompts")
    for fn in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
        if fn.endswith(".md"):
            fp = os.path.join(pdir, fn)
            with open(fp, encoding="utf-8") as fh:
                t = fh.read()
            with open(fp, "w", encoding="utf-8") as fh:
                fh.write(rewrite_paths(t))

    # bring the shared scripts this pipeline needs into its own scripts/
    sdir = os.path.join(dst, "scripts")
    os.makedirs(sdir, exist_ok=True)
    for s in spec["extra_scripts"]:
        shutil.copy2(os.path.join(CC_SCRIPTS, s), os.path.join(sdir, s))

    # bring the shared resource docs into the skill root (with path rewrites, so
    # their own references to verify_provenance.py etc. point at this skill's scripts/)
    for d in spec["docs"]:
        with open(os.path.join(CC_ROOT, d), encoding="utf-8") as fh:
            t = fh.read()
        with open(os.path.join(dst, d), "w", encoding="utf-8") as fh:
            fh.write(rewrite_paths(t))

    return {
        "scripts": sorted(os.listdir(sdir)),
        "docs": [d for d in spec["docs"]],
        "prompts": len([f for f in os.listdir(pdir)]) if os.path.isdir(pdir) else 0,
    }


ROUTE_DESC = (
    "Decide which PK/PE curation pipelines apply to a paper. First classifies the "
    "paper as PK / PE / Both / Neither from its title + abstract, then selects the "
    "matching pipelines — pk_* for a PK paper, pe_* for a PE paper, both for Both, "
    "and none for Neither — and writes the selected pipeline skills to trigger next. "
    "Use after pk-pe-prepare, when the user asks which pipelines to run on a paper. "
    "Does NOT curate; returns an empty selection for non-PK/PE papers."
)

# Flat-layout dispatch map: a pipeline label (PipelineTypeEnum value) -> the
# ollama skill NAME to trigger. Distinct from the Claude bundle's map (which emits
# `pipelines/<dir>` paths); here the targets are first-class top-level skills.
ROUTE_MAP_SCRIPT = '''#!/usr/bin/env python3
"""Deterministic map from a pipeline label to the ollama curation skill to trigger.

The pk-pe-route skill's design stage emits a list of pipeline labels (the
values of `extractor.constants.PipelineTypeEnum`). Turning those into the skills to
trigger must NOT be left to the model: the naming is irregular (`pk_summary` ->
`pk-summary-curation`, but `pk_drug_summary` -> `pk-drug-summary`), so a string
transform would silently misroute. This table is the single source of truth.

CLI: turn a selection into the `selected_pipelines.json` artifact.

    python pipeline_skill_map.py --pmid 12345678 --paper-type Both \\
        pk_summary pe_study_outcome > selected_pipelines.json
"""
import argparse
import json
import sys

# label (PipelineTypeEnum value) -> ollama skill name to trigger next.
PIPELINE_TO_SKILL = {
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

# domain of each label, for the paper-type gate.
PK_LABELS = {k for k in PIPELINE_TO_SKILL if k.startswith("pk_")}
PE_LABELS = {k for k in PIPELINE_TO_SKILL if k.startswith("pe_")}


def resolve(labels):
    """Map pipeline labels -> [{pipeline, skill}], preserving order, de-duped.

    Raises KeyError on an unknown label (a misspelled / hallucinated pipeline).
    """
    seen = set()
    out = []
    for label in labels:
        label = label.strip()
        if not label or label in seen:
            continue
        if label not in PIPELINE_TO_SKILL:
            raise KeyError(
                f"unknown pipeline label {label!r}; valid: {sorted(PIPELINE_TO_SKILL)}"
            )
        seen.add(label)
        out.append({"pipeline": label, "skill": PIPELINE_TO_SKILL[label]})
    return out


def build_selection(pmid, paper_type, labels):
    return {
        "pmid": pmid,
        "paper_type": paper_type,
        "selected": resolve(labels),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("labels", nargs="*", help="selected pipeline labels (PipelineTypeEnum values)")
    ap.add_argument("--pmid", default=None)
    ap.add_argument("--paper-type", default=None, help="PK / PE / Both / Neither")
    ap.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    args = ap.parse_args(argv)

    try:
        selection = build_selection(args.pmid, args.paper_type, args.labels)
    except KeyError as e:
        sys.exit(str(e))

    text = json.dumps(selection, indent=2, ensure_ascii=False) + "\\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
'''


def build_route_skill() -> None:
    """Standalone selector skill: identify paper type, then pick the pipelines.

    Ports the Claude bundle's route stage (pipelines/route/) — PKPEIdentificationStep
    + PKPEDesignStep — as a first-class triggerable skill that writes the selection
    but does NOT curate or dispatch.
    """
    name = "pk-pe-route"
    dst = os.path.join(DST, name)
    os.makedirs(os.path.join(dst, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(dst, "prompts"), exist_ok=True)

    # flat-name dispatch map (generated, not the Claude bundle's path-based one)
    with open(os.path.join(dst, "scripts", "pipeline_skill_map.py"), "w", encoding="utf-8") as fh:
        fh.write(ROUTE_MAP_SCRIPT)

    # route prompts (identify + design), with the map-script path rewritten to local
    for fn in ("01_identify.md", "02_design.md"):
        src = os.path.join(SRC, "pipelines", "route", "prompts", fn)
        with open(src, encoding="utf-8") as fh:
            t = fh.read()
        t = rewrite_paths(t)
        t = t.replace("pipelines/route/scripts/pipeline_skill_map.py", "scripts/pipeline_skill_map.py")
        with open(os.path.join(dst, "prompts", fn), "w", encoding="utf-8") as fh:
            fh.write(t)

    body = f"""# PK/PE Route

The **selector** of the curation suite. It ports two legacy steps —
`PKPEIdentificationStep` (PK / PE / Both / Neither) and `PKPEDesignStep`
(multi-label pipeline selection) — and writes the list of curation skills that
apply to the paper. It **does not curate**: hand the selection to the matching
pipeline skills (or trigger them yourself), one at a time.

{PATH_NOTE}
{SCRATCH_NOTE}
## Prerequisite
Run **pk-pe-prepare** first to produce `./.paper_assets/<pmid>/` (`paper_text.md`,
`abstract.md`, `table_<n>.md` / `table_<n>.html`, `manifest.json`). If the user
pasted raw title / abstract / full text, you can work from that directly.

## Scratch directory
Write intermediates to `./.pk_pe_route_scratch/<pmid>/` in the user's project /
working dir (never inside the skill folder; it is git-ignored): `identify.json`,
`design.json`, and the final `selected_pipelines.json`.

## Workflow
1. **Stage 1 — Identify** (`prompts/01_identify.md`): from the title + `abstract.md`,
   classify the paper as **PK / PE / Both / Neither** → `identify.json`.
2. **Paper-type gate** — the classification fixes the candidate set:
   - **PK** → choose only from the **PK** pipelines (`pk_*`).
   - **PE** → choose only from the **PE** pipelines (`pe_*`).
   - **Both** → choose from **both** `pk_*` and `pe_*`.
   - **Neither** → select **nothing**: write an empty `selected_pipelines.json`
     (`{{"pmid": "<pmid>", "paper_type": "Neither", "selected": []}}`), tell the user
     the paper is out of scope, and **stop** (do not run Stage 2).
3. **Stage 2 — Design** (`prompts/02_design.md`): within the gated candidate set,
   reconstruct the full text with tables visible (splice each `table_<n>.md` at its
   `[Table N]` marker) and select the **union** of all applicable pipelines
   (multi-label, non-exclusive) → `design.json`.
4. **Deterministic dispatch map** — never hand-write the skill names; run the
   byte-stable table:
   ```bash
   OUT="${{SKILL_SCRATCH_FOLDER:-.}}"; mkdir -p "$OUT/.pk_pe_route_scratch/<pmid>"
   python scripts/pipeline_skill_map.py \\
       --pmid <pmid> --paper-type <PK|PE|Both> <pipeline_tools from design.json> \\
       > "$OUT/.pk_pe_route_scratch/<pmid>/selected_pipelines.json"
   ```

## Candidate pipelines
- **PK** (`pk_*`): `pk_summary`, `pk_individual`, `pk_specimen_summary`,
  `pk_specimen_individual`, `pk_drug_summary`, `pk_drug_individual`,
  `pk_population_summary`, `pk_population_individual`.
- **PE** (`pe_*`): `pe_study_info`, `pe_study_outcome`.

## Output — `selected_pipelines.json`
```json
{{ "pmid": "12345678", "paper_type": "Both",
  "selected": [
    {{"pipeline": "pk_summary", "skill": "pk-summary-curation"}},
    {{"pipeline": "pe_study_outcome", "skill": "pe-study-outcome"}} ] }}
```
Each `skill` is the name of the standalone curation skill to trigger next. For a
`Neither` paper, `selected` is `[]`.

## Notes
- **Tables are visible to the design stage by design.** The legacy step saw table
  data inline in the full text; Stage 2 reconstructs that by splicing the
  `table_<n>.md` files back at their `[Table N]` markers. Do not run the design
  stage on the bare `paper_text.md` (markers only).
- The **label→skill map is deterministic** (`scripts/pipeline_skill_map.py`); only
  the identify + design judgements are model-driven.
"""
    fm = f"---\nname: {name}\ndescription: {ROUTE_DESC}\n---\n\n"
    with open(os.path.join(dst, "SKILL.md"), "w", encoding="utf-8") as fh:
        fh.write(fm + body)


def build_prepare_skill() -> None:
    """Standalone front-door skill: prepare a paper (HTML or JATS/PMC XML)."""
    name = "pk-pe-prepare"
    dst = os.path.join(DST, name)
    os.makedirs(os.path.join(dst, "scripts"), exist_ok=True)
    shutil.copy2(os.path.join(ASSETS, "prepare_paper.py"), os.path.join(dst, "scripts", "prepare_paper.py"))
    shutil.copy2(os.path.join(CC_SCRIPTS, "requirements.txt"), os.path.join(dst, "scripts", "requirements.txt"))

    desc = (
        "Prepare a PK/PE paper for curation: convert a publisher HTML file or a "
        "JATS/PMC XML file into the canonical input layout (paper_text.md with "
        "references stripped and tables replaced by [Table N] markers, abstract.md, "
        "per-table table_<n>.md / table_<n>.html, and manifest.json). Use this first, "
        "before the curation skills, when the user has a raw paper file (.html or .xml)."
    )
    body = f"""# PK/PE Prepare

The deterministic **front door** of the curation suite. It turns one raw paper
file — **HTML** (PMC / Wiley / Elsevier) or **JATS/PMC XML** — into the canonical
input layout every curation skill expects. Format is auto-detected from the file
extension and root element.

{PATH_NOTE}
{SCRATCH_NOTE}
## Run
```bash
# output base: $SKILL_SCRATCH_FOLDER if set, else the current directory
OUT="${{SKILL_SCRATCH_FOLDER:-.}}"
python scripts/prepare_paper.py <paper.html|paper.xml> --out "$OUT/.paper_assets"
# a directory of .html/.xml files works too:
python scripts/prepare_paper.py <dir> --out "$OUT/.paper_assets"
python scripts/prepare_paper.py <paper> --dry-run     # report only, write nothing
```
Needs `beautifulsoup4` **only for HTML** input (`pip install -r scripts/requirements.txt`);
the XML path is Python-3 standard library only.

## Output — `./.paper_assets/<pmid>/`
| File | Contents |
|---|---|
| `paper_text.md` | title (H1) + body as Markdown; references stripped; each data table → a `[Table N]` marker |
| `abstract.md` | the abstract as Markdown |
| `table_<n>.md` | table *n*'s caption + footnotes |
| `table_<n>.html` | table *n* as a `<section>` (caption + table + footnotes) |
| `manifest.json` | title, table count, and the `[Table N]` ↔ file mapping |

Tables are numbered by order of appearance. `<pmid>` is the input file's base name.

## Hand off
Point the curation skills at the produced `<pmid>/` directory:
- **table** skills (`pk-summary-curation`, `pk-individual-curation`,
  `pe-study-outcome`) → `table_<n>.html`;
- **full-text** skills (`pk-drug-*`, `pk-specimen-*`, `pk-population-*`,
  `pe-study-info`) → `paper_text.md` (+ `abstract.md`).

## Scope & notes
- **HTML** uses best-effort PMC / Wiley / Elsevier selectors. **XML** uses the
  NLM/JATS `<article>` schema (`<front>` metadata, `<body>` sections,
  `<table-wrap>` tables, `<ref-list>` references).
- Output is written to a git-ignored scratch dir in the user's project root
  (`./.paper_assets/`), never inside this skill folder.
- If the user pasted raw text instead of a file, place it into the equivalent
  `paper_text.md` / `table_<n>.html` files by hand.
"""
    fm = f"---\nname: {name}\ndescription: {desc}\n---\n\n"
    with open(os.path.join(dst, "SKILL.md"), "w", encoding="utf-8") as fh:
        fh.write(fm + body)


def write_install() -> None:
    txt = """# Installing the Ollama PK/PE curation skills

This `ollama_skills/` bundle is the **flat, self-contained** variant of the
PK/PE curation suite, tuned for smaller open models (e.g. Qwen via Ollama) run
through Claude Code. Each curation pipeline is its **own** top-level skill with
its **own** scripts and resources — there is no shared `curation-common/`.

## Install
Copy every folder in here into your project's `.claude/skills/`:

```bash
cp -R ollama_skills/* <your-project>/.claude/skills/
```

You then have 12 skills: the `pk-pe-prepare` front door, the
`pk-pe-route` selector, plus 10 standalone curation skills. There is
no router skill — full orchestration is the Claude bundle's job and is too much for
small open models; here you prepare the paper, optionally ask which pipelines
apply, then trigger each pipeline skill yourself.

## Use
- **Prepare first (any path):** trigger `pk-pe-prepare` (or run its
  `scripts/prepare_paper.py`) on a raw `.html` **or** `.xml` paper to produce
  `./.paper_assets/<pmid>/`.
- **Pick pipelines (optional):** trigger `pk-pe-route` to classify the
  paper (PK / PE / Both / Neither) and get the list of applicable pipeline skills in
  `selected_pipelines.json`. Returns an empty list for non-PK/PE papers.
- **Then curate (one pipeline at a time):** trigger each selected pipeline skill,
  e.g. "use pk-individual-curation to curate paper <pmid>", pointing it at the
  prepared `./.paper_assets/<pmid>/` files. Triggering a single skill keeps its full
  procedure in front of the model — the reliable path for the smallest models.

## Where files go (two env vars)
Each skill writes two kinds of files; both default to the user's current working
directory and can be redirected with an environment variable:

- **Intermediate / working files** — the prepared `.paper_assets/` and the
  per-skill `.<name>_scratch/` dirs (git-ignored). Set **`SKILL_SCRATCH_FOLDER`**
  to root them elsewhere.
- **Final results** — each curation skill copies its deliverable CSV to
  `<pmid>/<skill>.csv` as its last step. Set **`SKILL_OUTPUT_FOLDER`** to root those
  elsewhere (e.g. one clean results dir, separate from the noisy scratch dirs).

```bash
export SKILL_SCRATCH_FOLDER=/tmp/pkpe-work       # intermediates
export SKILL_OUTPUT_FOLDER=/data/pkpe-results    # final CSVs
```

Unset (the default), both go in the current directory. The two are independent —
set either, both, or neither.

## Dependencies
Pipelines that convert HTML tables or prepare papers need BeautifulSoup:
`pip install -r <skill>/scripts/requirements.txt` (provides `beautifulsoup4`).
`verify_provenance.py` and the `clean_*_rows.py` cleanup scripts are Python-3
stdlib only.

## Regenerating
Do not hand-edit this folder. It is generated from `skills/pk-pe-curation/` by:

```bash
python scripts/build_ollama_skills.py
```
"""
    with open(os.path.join(DST, "INSTALL.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)


def main() -> int:
    if not os.path.isdir(SRC):
        print(f"source not found: {SRC}", file=sys.stderr)
        return 1
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)

    print(f"Generating {os.path.relpath(DST, REPO)}/ from {os.path.relpath(SRC, REPO)}/\n")
    for name, spec in PIPELINES.items():
        info = copy_pipeline(name, spec)
        print(f"  {name}")
        print(f"      scripts: {info['scripts']}")
        print(f"      docs:    {info['docs']}    prompts: {info['prompts']}")
    build_prepare_skill()
    print("  pk-pe-prepare (front door, HTML + XML)")
    build_route_skill()
    print("  pk-pe-route (selector)")
    write_install()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
