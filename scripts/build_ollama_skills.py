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
  * `pk-pe-curation` remains as a THIN ROUTER that prepares the paper, decides
    which pipelines apply, and dispatches by TRIGGERING the pipeline skills.

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

# --- Per-pipeline manifest -------------------------------------------------
# `extra_scripts` are pulled from curation-common/scripts/ into the skill's own
# scripts/. Scripts already living in the pipeline's own scripts/ (e.g.
# clean_individual_rows.py) are copied verbatim with the pipeline and need no
# entry here. `docs` are resource .md files pulled from curation-common/ into
# the skill root.
PIPELINES = {
    "pk-summary-curation": {
        "desc": "Curate aggregate/summary pharmacokinetics (PK) tables (mean / median / SD / range across a cohort) from a biomedical paper into a normalized 19-column dataset. Use for summary PK tables; for per-subject tables use pk-individual-curation. Runs standalone or dispatched by the pk-pe-curation router.",
        "extra_scripts": ["verify_provenance.py", "html_to_markdown_table.py", "requirements.txt"],
        "docs": ["verify_and_correct.md"],
    },
    "pk-individual-curation": {
        "desc": "Curate individual-subject pharmacokinetics (PK) tables (one row per subject per parameter) from a biomedical paper into a normalized 12-column dataset. Use when tables report per-individual PK values; do NOT use for summary/aggregate tables (use pk-summary-curation). Runs standalone or dispatched by the pk-pe-curation router.",
        "extra_scripts": ["verify_provenance.py", "html_to_markdown_table.py", "requirements.txt"],
        "docs": ["verify_and_correct.md"],
    },
    "pe-study-outcome": {
        "desc": "Curate pharmacoepidemiology (PE) study-outcome tables from a paper into a normalized 12-column dataset (effect estimates, confidence intervals, p-values per outcome). Runs standalone or dispatched by the pk-pe-curation router.",
        "extra_scripts": ["verify_provenance.py", "html_to_markdown_table.py", "clean_pe_outcome_rows.py", "requirements.txt"],
        "docs": ["verify_and_correct.md"],
    },
    "pe-study-info": {
        "desc": "Extract pharmacoepidemiology (PE) study metadata (design, population, exposure, outcome definitions) from a paper's full text into a 10-column single-row dataset. Runs standalone or dispatched by the pk-pe-curation router.",
        "extra_scripts": ["verify_provenance.py"],
        "docs": ["verify_and_correct.md"],
    },
    "pk-drug-summary": {
        "desc": "Extract summary (cohort-level) drug dosing regimens (dose amount / unit / frequency / route) from a PK paper's full text into an 11-column dataset. For per-patient dosing use pk-drug-individual. Runs standalone or dispatched by the pk-pe-curation router.",
        "extra_scripts": ["verify_provenance.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
    "pk-drug-individual": {
        "desc": "Extract per-patient drug dosing regimens from a PK paper's full text into an 11-column dataset. For cohort-level dosing use pk-drug-summary. Runs standalone or dispatched by the pk-pe-curation router.",
        "extra_scripts": ["verify_provenance.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
    "pk-population-summary": {
        "desc": "Extract summary population/demographic characteristics with statistics from a PK paper's full text into a 15-column dataset. For per-patient characteristics use pk-population-individual. Runs standalone or dispatched by the pk-pe-curation router.",
        "extra_scripts": ["verify_provenance.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
    "pk-population-individual": {
        "desc": "Extract per-patient population/demographic characteristics from a PK paper's full text into a 9-column dataset. For cohort-level stats use pk-population-summary. Runs standalone or dispatched by the pk-pe-curation router.",
        "extra_scripts": ["verify_provenance.py", "clean_population_individual_rows.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
    "pk-specimen-summary": {
        "desc": "Extract summary (cohort-level) specimen-sampling information (specimen type, sampling times) from a PK paper's full text into a 9-column dataset. For per-patient sampling use pk-specimen-individual. Runs standalone or dispatched by the pk-pe-curation router.",
        "extra_scripts": ["verify_provenance.py", "clean_specimen_rows.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
    "pk-specimen-individual": {
        "desc": "Extract per-patient specimen-sampling information from a PK paper's full text into a 9-column dataset. For cohort-level sampling use pk-specimen-summary. Runs standalone or dispatched by the pk-pe-curation router.",
        "extra_scripts": ["verify_provenance.py", "clean_specimen_rows.py"],
        "docs": ["verify_and_correct.md", "refine_population.md"],
    },
}

ROUTER_DESC = (
    "Curate pharmacokinetics (PK) and pharmacoepidemiology (PE) data from a biomedical "
    "paper into normalized CSV datasets. Entry point that prepares the paper, decides "
    "which curation pipelines apply, and dispatches to them by triggering the matching "
    "self-contained pipeline skill. Use when the user has a PK/PE paper (HTML, or pasted "
    "full text + tables) and wants structured data extracted. Do NOT use for non-PK/PE papers."
)

PATH_NOTE = (
    "> **Self-contained skill.** Every `scripts/…` path and every resource file "
    "(`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this "
    "skill's own directory**. This skill shares nothing with other skills — when run as "
    "an installed skill, resolve these paths under this skill's folder.\n"
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


def make_skill_md(name: str, desc: str, body: str) -> str:
    body = rewrite_paths(body)
    fm = f"---\nname: {name}\ndescription: {desc}\n---\n\n"
    return fm + PATH_NOTE + "\n" + body


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


def build_router() -> None:
    name = "pk-pe-curation"
    dst = os.path.join(DST, name)
    os.makedirs(os.path.join(dst, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(dst, "prompts"), exist_ok=True)

    # router-owned scripts
    shutil.copy2(os.path.join(CC_SCRIPTS, "prepare_paper.py"), os.path.join(dst, "scripts", "prepare_paper.py"))
    shutil.copy2(os.path.join(CC_SCRIPTS, "requirements.txt"), os.path.join(dst, "scripts", "requirements.txt"))
    shutil.copy2(
        os.path.join(SRC, "pipelines", "route", "scripts", "pipeline_skill_map.py"),
        os.path.join(dst, "scripts", "pipeline_skill_map.py"),
    )
    # route prompts (identify + design)
    for fn in ("01_identify.md", "02_design.md"):
        src = os.path.join(SRC, "pipelines", "route", "prompts", fn)
        with open(src, encoding="utf-8") as fh:
            t = fh.read()
        with open(os.path.join(dst, "prompts", fn), "w", encoding="utf-8") as fh:
            fh.write(rewrite_paths(t))

    table_skills = "pk-summary-curation, pk-individual-curation, pe-study-outcome"
    body = f"""# PK/PE Curation — router

The entry point for extracting structured PK/PE data from a paper. It runs three
stages: **prepare** the paper, **route** (decide which pipelines apply), then
**dispatch**. Unlike the Claude bundle, each curation pipeline here is a
**separate, self-contained skill** — this router does not contain them; it
**triggers** them.

{PATH_NOTE}
## Inputs
- A paper — ideally publisher **HTML** (PMC / Wiley / Elsevier), or pasted full
  text + tables. The title (and PMID) help with naming.

## Stage A — Prepare the paper
```bash
python scripts/prepare_paper.py <paper.html> --out ./.paper_assets
```
Produces `./.paper_assets/<pmid>/` with `paper_text.md` (references stripped,
tables → `[Table N]` markers), `abstract.md`, `table_<n>.md` / `table_<n>.html`,
and `manifest.json`. (HTML only; if the user pasted raw text, place it into the
equivalent files yourself.)

## Stage B — Route (which pipelines apply)
1. **Identify** (`prompts/01_identify.md`) — PK / PE / Both / Neither from title +
   `abstract.md`. **If `Neither`, stop** and report it.
2. **Design** (`prompts/02_design.md`) — select the union of applicable pipelines.
3. **Dispatch map**:
   ```bash
   python scripts/pipeline_skill_map.py \\
       --pmid <pmid> --paper-type <PK|PE|Both> <pipeline_tools...> \\
       > ./.pk_pe_route_scratch/<pmid>/selected_pipelines.json
   ```

## Stage C — Dispatch by TRIGGERING each pipeline skill
For **each** entry in `selected_pipelines.json`, **trigger the curation skill whose
name is the entry's `pipeline` value** (invoke it as a skill — do **not** read its
procedure files yourself), giving it the right input from `./.paper_assets/<pmid>/`:

- **Table skills** ({table_skills}) → the relevant `table_<n>.html`.
- **Full-text skills** (`pk-drug-*`, `pk-specimen-*`, `pk-population-*`,
  `pe-study-info`) → `paper_text.md` (and `abstract.md` for context).

**Important for smaller / Ollama models:** trigger **one** pipeline skill, let it
run its **entire** procedure and write its CSV, then trigger the next. Triggering a
pipeline loads its full instructions as active content — that is what keeps a
small model from skipping stages. Do **not** attempt to run the pipelines' internal
stages yourself from this router.

## The ten pipeline skills

| Skill | Source | Output |
|---|---|---|
| `pk-summary-curation` | table | 19-col aggregate PK |
| `pk-individual-curation` | table | 12-col per-subject PK |
| `pk-drug-summary` / `pk-drug-individual` | full text | 11-col dosing regimen |
| `pk-specimen-summary` / `pk-specimen-individual` | full text | 9-col specimen sampling |
| `pk-population-summary` | full text | 15-col demographics + stats |
| `pk-population-individual` | full text | 9-col per-patient characteristics |
| `pe-study-info` | full text | 10-col study metadata (single row) |
| `pe-study-outcome` | table | 12-col PE outcomes |

## Notes
- **Scratch state** is written to git-ignored dirs in the user's project root
  (`./.paper_assets/`, `./.pk_pe_route_scratch/`, each pipeline's
  `./.{{pipeline}}_scratch/`), never inside a skill folder.
- Each pipeline skill is **self-contained** (its own `scripts/` + resources) and
  ends with a bounded verify/correct quality gate.
"""
    fm = f"---\nname: {name}\ndescription: {ROUTER_DESC}\n---\n\n"
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

You then have 11 skills: the `pk-pe-curation` router plus 10 standalone
curation skills.

## Use
- **Guided (router):** ask to "curate PK/PE data from paper <pmid>" — the
  `pk-pe-curation` router prepares + routes, then triggers the matching pipeline
  skills one at a time.
- **Direct (recommended for the smallest models):** trigger a single pipeline
  skill, e.g. "use pk-individual-curation to curate paper <pmid>", once you know
  which table type you have. This keeps the full procedure in front of the model.

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
    build_router()
    print("  pk-pe-curation (router)")
    write_install()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
