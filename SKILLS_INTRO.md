# Curation Skills

This repository contains a **single bundled Claude skill**, `pk-pe-curation`, that
re-implements the PK/PE curation pipelines as model-driven, prose-defined
procedures rather than hard-coded LangGraph workflows. It lives under
`skills/pk-pe-curation/` and is exercised by the regression fixtures in
`skills_e2e_tests/`.

The goal is to run the same curation logic under **Claude Code pointed at an
Ollama server** (Gemma / Qwen on OSC) as well as under Claude itself, and to be
able to compare model behavior across both. A skill is just a folder the model
reads — no Python runtime, no graph engine — so it ports across hosts as long
as the model can follow instructions and call the few bundled scripts. It is
installed by copying the one folder into `<project>/.claude/skills/` (see
`skills/pk-pe-curation/INSTALL.md`); Claude Code then auto-triggers it from the
top `SKILL.md`'s `description`.

## What the bundle is

`pk-pe-curation/` is one skill directory:
- **`SKILL.md`** — the single triggerable **orchestrator**: YAML front-matter
  (`name`, `description`), a path-anchor note, and the prose flow
  *prepare → route → dispatch*. It is the only file Claude Code registers as a
  skill.
- **`pipelines/`** — twelve sub-procedure directories (the ten curation pipelines
  plus `prepare-paper` and `route`). Each has a `procedure.md` (was its own
  `SKILL.md`; no front-matter — not separately triggerable) and a `prompts/`
  directory of per-stage markdown. A few also carry their own `scripts/`.
- **`curation-common/`** — the shared support library: deterministic stdlib/`bs4`
  scripts plus the generic verify/correct and population-refine prompts every
  pipeline reuses.

There is **no templating engine**. The prompt files are prose the model reads and
follows, not `{variable}` templates. State between stages is passed through files
on disk, not a graph's typed state object. **All bundled paths** (`curation-common/…`,
`pipelines/…`) are relative to the skill's own directory, so they resolve whether
the bundle sits at `skills/pk-pe-curation/` (repo) or `.claude/skills/pk-pe-curation/`
(installed).

## The pipelines (internal sub-procedures)

`procedure` paths below are relative to the bundle (`pipelines/<name>`). The
orchestrator follows `<procedure>/procedure.md` for each one it selects.

| Sub-procedure | Source | Purpose | Output |
|-------|--------|---------|--------|
| `prepare-paper` | HTML | **Front door** — split a raw paper into the canonical input layout (`paper_text.md`, `abstract.md`, `table_<n>.md/.html`, `manifest.json`). Deterministic; run first. | files |
| `route` | prepared paper | **Selector** — classify PK/PE/Both/Neither, then select the union of applicable pipelines. Emits `selected_pipelines.json`. | dispatch list |
| `pk-summary-curation` | table | Aggregate PK tables (mean / median / SD / range / CI of AUC, Cmax, t½, CL, Vd, …). | **19 columns** |
| `pk-individual-curation` | table | Per-subject PK tables (one row = one patient's own measured value). | **12 columns** |
| `pk-drug-summary` | full text | Dosing regimen (drug, dose, unit, frequency, schedule, route) per population group. | **11 columns** |
| `pk-drug-individual` | full text | Dosing regimen per individual patient/case. | **11 columns** |
| `pk-specimen-summary` | full text | Specimen sampling (specimen, sample count, sampling time) per population group. | **9 columns** |
| `pk-specimen-individual` | full text | Specimen sampling per individual patient/case. | **9 columns** |
| `pk-population-summary` | full text | Population characteristics (age, sex, weight, BMI, …) + summary statistics per group. | **15 columns** |
| `pk-population-individual` | full text | Per-patient characteristics (raw value). | **9 columns** |
| `pe-study-info` | full text | PE study-level metadata (study type/design, data source, population, criteria, outcomes). | **10 columns** (single row) |
| `pe-study-outcome` | table | PE outcome table — numeric outcomes tagged by characteristic/exposure/outcome with stats. | **12 columns** |

The user never selects a pipeline directly — Claude Code triggers the one
`pk-pe-curation` skill, whose orchestrator routes to the pipelines internally.

> All ten legacy PK/PE curation pipelines are ported (the two table PK pipelines,
> the six full-text PK pipelines, and the two PE pipelines). `pe-study-outcome`
> implements the active `pe_study_outcome_ver2`; the deprecated v1 (header/row
> categorize + mapping) is intentionally not ported.

### Front door: `prepare-paper` and the canonical input layout

The ten curation pipelines all assume their inputs already exist (clean full text,
isolated tables). `prepare-paper` produces them. It is a **deterministic**
(bs4-only, no model) step that turns one raw paper HTML into a per-paper directory
under `./.paper_assets/<pmid>/`:

```
paper_text.md   # title (H1) + body; references stripped; tables → [Table N] markers
abstract.md
table_<n>.md    # caption + footnotes
table_<n>.html  # the table grid, as a <section>
manifest.json   # title, table count, and the [Table N] ↔ table_<n>.* index
```

This is the **canonical input contract** for the whole suite: full-text pipelines
read `paper_text.md`; table pipelines read the isolated `table_<n>.html`; and the
`route` selector reads `paper_text.md` + `manifest.json` (splicing each
`table_<n>.md` back at its `[Table N]` marker to recover the legacy "full text with
tables visible" view the design step relied on). Scope is **HTML only** for now —
JATS/PMC XML is a planned follow-up.

### Selecting pipelines: `route`

`prepare-paper` gives you the files; `route` decides which of the ten curation
pipelines to run on them. It ports two legacy steps — `PKPEIdentificationStep`
(PK/PE/Both/Neither, from title + abstract) and `PKPEDesignStep` (multi-label,
non-exclusive pipeline selection, from the tables-visible full text) — and emits
`selected_pipelines.json`, a deterministic dispatch list mapping each selected
pipeline label to its **sub-procedure** (`pipelines/<name>`). The label→procedure
map is byte-tested (`test_pipeline_skill_map.py`) because the naming is irregular
(`pk_summary` → `pk-summary-curation`, but `pk_drug_summary` → `pk-drug-summary`).
By default the orchestrator reports the selection and confirms before running each
pipeline — robust on small Ollama models; full auto-orchestration is an opt-in.

### Two pipeline shapes: table-driven vs full-text-driven
- **Table pipelines** (`pk-summary-curation`, `pk-individual-curation`,
  `pe-study-outcome`) take **tables** (HTML or markdown). They run Stage 0
  (HTML→Markdown via the shared converter); the PK pair also run Stage 0b (select
  the PK tables when several are supplied) and curate each in its own `table_<n>/`
  sub-directory. `pe-study-outcome` curates the single outcome table it is given.
- **Full-text pipelines** (`pk-drug-summary`, `pk-drug-individual`,
  `pk-specimen-summary`, `pk-specimen-individual`, `pk-population-summary`,
  `pk-population-individual`, `pe-study-info`) take the paper's **full text**
  instead of a table. There is no Stage 0 conversion, no table
  selection, and no `table_<n>/` nesting — the scratch dir is flat (one run = one
  output). They share a 5-stage skeleton: *extract-from-prose → patient/population
  refine → domain refine → assemble → verify*. The patient/population-refine step
  is identical across them, so it lives once in
  `curation-common/refine_population.md`. Their verify stage runs the provenance
  script in **existence-only** mode (the source is prose, so the table-based
  attribution check does not apply).

### Key differences between the two pipelines

The two pipelines mirror each other but are not the same:
- **pk-individual** keys on a per-row `Patient ID` (not a `Subject N` count),
  has **no** statistic / variation / interval / P-value block (each row is one
  raw value), and **inverts** the summary-vs-individual row deletion (it keeps
  the per-subject rows).
- **pk-individual Stage 0c** infers and injects a `Patient ID` from the paper's
  **full text** when the table itself has none — essential for single-patient
  case reports, where the subject's identity lives in the prose. Without it such
  a table would be discarded for lacking a Patient ID. This is why
  pk-individual takes the full text as an input and pk-summary does not.
- The parameter-type transpose condition, the header-category set (3 vs 5
  categories), and the per-stage column lists differ accordingly.

## Scratch-state model

A long curation run can have its conversation context summarized, which would
corrupt the exact table text later stages depend on. So **every stage writes its
output to a file and reads its inputs back from disk** — the conversation is
never the source of truth for intermediate tables.

- Summary runs use `./.pk_curation_scratch/<pmid>/`;
  individual runs use `./.pk_individual_scratch/<pmid>/`.
- The scratch directory is created in the **user's current project / working
  directory**, never inside the skill folder (the skill folder is read-only
  content). Both are git-ignored.
- The input may hold several tables. Each selected table gets its own
  `table_<n>/` sub-directory holding its full per-stage run; cross-table files
  (the all-tables listing, the selection note, the combined result) live at the
  top level. The pipeline **always nests, even for a single table** — one code
  path, fewer mistakes.

## Procedure shape

Both pipelines share the same top-level loop:
1. **Stage 0a** — convert every input table (HTML or markdown) to markdown and
   list them, using the bundled `html_to_markdown_table.py` (deterministic
   colspan/rowspan/multi-row-header handling — never parse HTML by hand).
2. **Stage 0b** — if there is more than one table, select the PK tables;
   otherwise skip the judgment.
3. **For each selected table** — run the ordered numbered stages (1–14 for
   summary; a 0c preprocess + 1–14 for individual) in its `table_<n>/`.
4. **Combine** — vertically concatenate each table's final CSV into
   `combined_final.csv`, the deliverable.

The final numbered stage of each pipeline is a **bounded verify → correct
quality gate** (`curation-common/verify_and_correct.md`): a deterministic
provenance + attribution check (`verify_provenance.py`) plus an adversarial
semantic review, then at most **2 correction rounds**. It never silently drops a
row to pass, and surfaces unresolved findings to the user.

## Deterministic vs semantic

The pieces that *can* be made deterministic are isolated into scripts and tested
byte-exactly in CI; everything else is model-driven and evaluated against
semantic oracles by hand or in an eval harness.

- **Deterministic (byte-exact in CI):** HTML→Markdown conversion
  (`html_to_markdown_table.py`), the provenance/attribution checker
  (`verify_provenance.py`), the pk-individual row cleanup
  (`clean_individual_rows.py`), and the Stage-0b selection-case *structure*.
- **Semantic (oracle-checked, not byte-exact):** every LLM stage — drug/patient
  extraction, alignment, splitting, value/time extraction, the selection
  judgment itself, and the verify/correct reasoning.

`verify_provenance.py` is schema-agnostic — it checks **numbers**, not column
names. Its *existence* check catches hallucinated/mistyped values; its
*attribution* check (argmax-Jaccard label matching) catches values copied to the
wrong row / cohort / specimen even when the number does appear somewhere in the
source.

## Testing

All skill regression tests live in `skills_e2e_tests/` and are deterministic
(no model), safe for CI:

```bash
poetry run pytest skills_e2e_tests                                    # everything
poetry run pytest skills_e2e_tests/test_stage0_conversion.py -v       # HTML→Markdown
poetry run pytest skills_e2e_tests/test_provenance.py -v              # provenance + attribution
poetry run pytest skills_e2e_tests/test_clean_individual_rows.py -v   # pk-individual cleanup
poetry run pytest skills_e2e_tests/test_table_selection.py -v         # Stage 0b selection structure
```

Fixtures bundle a real paper's source table plus the expected output of the
early stages, so prompt/script drift is detectable. See
`skills_e2e_tests/README.md` for the case layout, the two kinds of oracle, how
to evaluate the model-driven stages by hand, and the documented gap (a
pk-individual semantic case is not yet bundled).

## Relationship to the legacy pipeline

These skills are a re-expression of the LangGraph pipelines in `extractor/`
(see `CLAUDE.md` and the project memory). The legacy code remains the source of
truth for behavior, and `benchmark/` remains the source of truth for **accuracy**
— the skills deliberately do **not** score themselves against a gold standard.
When porting a stage, the pattern is: read the legacy agent in
`extractor/agents/<pipeline>/`, translate it to a prose prompt + scratch file,
and verify deterministically wherever the stage allows it.
