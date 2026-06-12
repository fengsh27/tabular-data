---
name: pk-pe-curation
description: Curate pharmacokinetics (PK) and pharmacoepidemiology (PE) data from a biomedical paper into normalized CSV datasets. Entry point that prepares the paper, decides which curation pipelines apply, and dispatches to them by triggering the matching self-contained pipeline skill. Use when the user has a PK/PE paper (HTML, or pasted full text + tables) and wants structured data extracted. Do NOT use for non-PK/PE papers.
---

# PK/PE Curation — router

The entry point for extracting structured PK/PE data from a paper. It runs three
stages: **prepare** the paper, **route** (decide which pipelines apply), then
**dispatch**. Unlike the Claude bundle, each curation pipeline here is a
**separate, self-contained skill** — this router does not contain them; it
**triggers** them.

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

## Inputs
- A paper — publisher **HTML** (PMC / Wiley / Elsevier) **or JATS/PMC XML**, or
  pasted full text + tables. The title (and PMID) help with naming.

## Stage A — Prepare the paper
```bash
python scripts/prepare_paper.py <paper.html|paper.xml> --out ./.paper_assets
```
(Or trigger the standalone **pk-pe-prepare** skill, which wraps the same script.)
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
   python scripts/pipeline_skill_map.py \
       --pmid <pmid> --paper-type <PK|PE|Both> <pipeline_tools...> \
       > ./.pk_pe_route_scratch/<pmid>/selected_pipelines.json
   ```

## Stage C — Dispatch by TRIGGERING each pipeline skill
For **each** entry in `selected_pipelines.json`, **trigger the curation skill whose
name is the entry's `pipeline` value** (invoke it as a skill — do **not** read its
procedure files yourself), giving it the right input from `./.paper_assets/<pmid>/`:

- **Table skills** (pk-summary-curation, pk-individual-curation, pe-study-outcome) → the relevant `table_<n>.html`.
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
  `./.{pipeline}_scratch/`), never inside a skill folder.
- Each pipeline skill is **self-contained** (its own `scripts/` + resources) and
  ends with a bounded verify/correct quality gate.
