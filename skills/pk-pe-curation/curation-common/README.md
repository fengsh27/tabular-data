# Curation Common

Shared building blocks for the table-curation skills. Each curation skill
(e.g. `pk-summary-curation`, `pk-individual-curation`) runs its own ordered
stages, but they all lean on the same two deterministic scripts and the same
verify/correct procedure, kept here once so they don't drift apart.

## Scripts (`scripts/`)
- **`html_to_markdown_table.py`** — Stage 0a converter. Turns one HTML `<table>`
  into a clean markdown table (colspan/rowspan expansion, multi-row header
  stacking, empty/duplicate-column cleanup). Self-contained except for
  `beautifulsoup4` (see `scripts/requirements.txt`). One `<table>` per
  invocation:
  ```
  python curation-common/scripts/html_to_markdown_table.py <path-to-html>
  ```
- **`verify_provenance.py`** — deterministic provenance check used by the
  verify/correct stage. Schema-agnostic: it checks **numbers**, not column
  names. Two checks:
  - *existence* (default) — every checked number appears in the source;
  - *attribution* (`--attribution --label-columns ...`) — each number sits
    under the source label (column header / row label) the curated row best
    matches, catching values copied to the wrong row/cohort/specimen.
  ```
  python curation-common/scripts/verify_provenance.py FINAL_CSV SOURCE [SOURCE ...] \
    --value-columns "<numeric cols>" \
    --attribution --label-columns "<discriminator cols>" --json
  ```
  Exit 0 clean, 1 if any finding, 2 on usage/IO error.

  **Table vs. full-text source.** The *attribution* mode parses **markdown
  tables** out of the source, so it only helps when the source is a table
  (the table-driven skills). For the **full-text** skills (pk-drug, pk-specimen,
  pk-population, pe-study-info) the source is prose, not a table — run the
  existence check **only** (`--value-columns`, *no* `--attribution`); attribution
  has nothing to match against and adds noise.

## Generic prompts
- **`verify_and_correct.md`** — the bounded verify→correct quality gate, written
  so any curation skill can reuse it by supplying its own `--value-columns` and
  `--label-columns` (and choosing existence-only vs. attribution per the
  table-vs-full-text note above). Each pipeline's final verification stage points
  here and fills in its column lists and scratch paths.
- **`refine_population.md`** — the shared patient/population-demographics
  refinement step used by every full-text skill. Normalizes `Population`,
  `Pregnancy stage`, and `Pediatric/Gestational age` into canonical categories,
  parameterized by a `<KEY-COLUMN>` (`Patient ID` for individual pipelines,
  `Population N` for summary pipelines) so one prose file serves both variants.

## How the pipeline skills reference this
The pipeline skills name these paths directly (e.g.
`curation-common/scripts/verify_provenance.py`). When adding a new
curation skill, reuse these rather than copying — fixes and hardening then apply
to every pipeline at once.
