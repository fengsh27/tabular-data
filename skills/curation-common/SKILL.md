---
name: curation-common
description: Shared, schema-agnostic tooling for the PK/PE table-curation skills
  (pk-summary-curation, pk-individual-curation, …). Holds the bundled scripts and
  the generic verify-and-correct procedure they all reuse. This is a SUPPORT
  library referenced by the other curation skills via file path — it is not meant
  to be invoked on its own; do not select it directly in response to a user
  request. Use the specific pipeline skill instead.
---

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
  python skills/curation-common/scripts/html_to_markdown_table.py <path-to-html>
  ```
- **`verify_provenance.py`** — deterministic provenance check used by the
  verify/correct stage. Schema-agnostic: it checks **numbers**, not column
  names. Two checks:
  - *existence* (default) — every checked number appears in the source;
  - *attribution* (`--attribution --label-columns ...`) — each number sits
    under the source label (column header / row label) the curated row best
    matches, catching values copied to the wrong row/cohort/specimen.
  ```
  python skills/curation-common/scripts/verify_provenance.py FINAL_CSV SOURCE [SOURCE ...] \
    --value-columns "<numeric cols>" \
    --attribution --label-columns "<discriminator cols>" --json
  ```
  Exit 0 clean, 1 if any finding, 2 on usage/IO error.

## Generic verify/correct procedure
- **`verify_and_correct.md`** — the bounded verify→correct quality gate, written
  so any curation skill can reuse it by supplying its own `--value-columns` and
  `--label-columns`. Each pipeline's final verification stage points here and
  fills in its column lists and scratch paths.

## How the pipeline skills reference this
The pipeline skills name these paths directly (e.g.
`skills/curation-common/scripts/verify_provenance.py`). When adding a new
curation skill, reuse these rather than copying — fixes and hardening then apply
to every pipeline at once.
