---
name: pk-pe-curation
description: Curate pharmacokinetics (PK) and pharmacoepidemiology (PE) data from a
  biomedical paper into normalized CSV datasets. Use this when the user has a PK/PE
  paper (as HTML, or as pasted full text + tables) and wants structured data
  extracted — drug PK summary/individual tables, dosing regimens, specimen sampling,
  population/demographic characteristics, or PE study info/outcomes. It prepares the
  paper, decides which curation pipelines apply, and runs them. This is the single
  entry point for the whole PK/PE curation suite; the individual pipelines
  (pk-summary, pk-individual, pk-drug-*, pk-specimen-*, pk-population-*, pe-study-*)
  are internal sub-procedures it dispatches to. Do NOT use for non-PK/PE papers.
---

# PK/PE Curation

The single entry point for extracting structured PK/PE data from a paper. It runs
three stages: **prepare** the paper into clean inputs, **route** (decide which
pipelines apply), then **curate** with each selected pipeline. Ten curation
pipelines live inside this skill as sub-procedures under `pipelines/`; they share
the support library in `curation-common/`.

## Resolving bundled paths (read first)

All paths in this skill and its sub-procedures that begin with `curation-common/`
or `pipelines/` are **relative to this skill's own directory** — the
`pk-pe-curation/` folder you loaded this `SKILL.md` from. That folder is `skills/`
in the source repo and `.claude/skills/pk-pe-curation/` when installed in a user's
project. **Before running any bundled script or reading any bundled file, prefix
its path with this skill's actual directory.** For example, when installed:

```bash
python .claude/skills/pk-pe-curation/curation-common/scripts/prepare_paper.py ...
```

Do not run the bundled paths verbatim from the project root — they live under this
skill's directory, not under the user's CWD.

## Inputs you need
- **A paper** — ideally the publisher's **HTML** (PMC / Wiley / Elsevier). If the
  user pastes full text + tables directly, you can use that too.
- The paper's **title** (and ideally PMID) for naming and disambiguation.

This skill does not fetch papers or read PDFs — ask the user for the HTML or text.

## Stage A — Prepare the paper
Follow `pipelines/prepare-paper/procedure.md`. It runs the deterministic converter:

```bash
python curation-common/scripts/prepare_paper.py <paper.html> --out ./.paper_assets
```

producing `./.paper_assets/<pmid>/` with `paper_text.md` (references stripped,
tables → `[Table N]` markers), `abstract.md`, `table_<n>.md` / `table_<n>.html`,
and `manifest.json`. (HTML only; if the user pasted raw text, place it into the
equivalent files yourself.)

## Stage B — Route (which pipelines apply)
Follow `pipelines/route/procedure.md`:
1. **Identify** (`pipelines/route/prompts/01_identify.md`) — PK / PE / Both /
   Neither from title + `abstract.md`. **If `Neither`, stop** and report it.
2. **Design** (`pipelines/route/prompts/02_design.md`) — select the union of
   applicable pipelines from the tables-visible full text.
3. **Dispatch map** — turn the selection into `selected_pipelines.json`:
   ```bash
   python pipelines/route/scripts/pipeline_skill_map.py \
       --pmid <pmid> --paper-type <PK|PE|Both> <pipeline_tools...> \
       > ./.pk_pe_route_scratch/<pmid>/selected_pipelines.json
   ```
   Each entry is `{"pipeline": "...", "procedure": "pipelines/<name>"}`.

## Stage C — Curate with each selected pipeline
By **default, report the selected pipelines and confirm with the user** before
running them — this is the robust path under smaller (Ollama) models. When the user
wants an end-to-end run (or has already confirmed), for **each** entry in
`selected_pipelines.json` follow `<procedure>/procedure.md`, giving it the right
input from the prepared directory:

- **Table pipelines** — `pk-summary-curation`, `pk-individual-curation`,
  `pe-study-outcome` → the relevant `table_<n>.html`.
- **Full-text pipelines** — `pk-drug-*`, `pk-specimen-*`, `pk-population-*`,
  `pe-study-info` → `paper_text.md` (and `abstract.md` for context).

Each pipeline writes its own scratch dir and produces a final CSV in its declared
schema, ending with a bounded verify/correct quality gate. Collect the per-pipeline
CSVs as the deliverables.

## The ten pipelines (internal sub-procedures)

| `procedure` (under `pipelines/`) | Source | Output |
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
  (`./.paper_assets/`, `./.pk_pe_route_scratch/`, and each pipeline's
  `./.{pipeline}_scratch/`), never inside this skill folder.
- **Deterministic vs semantic:** the bundled scripts (`prepare_paper.py`,
  `pipeline_skill_map.py`, `html_to_markdown_table.py`, `verify_provenance.py`, the
  row-cleanup scripts) are deterministic and regression-tested in
  `skills_e2e_tests/`. The reasoning stages are model-driven and evaluated against
  semantic oracles.
