# Stage 06 — Header categorization

This file is loaded by `SKILL.md` during the PK individual curation procedure.

## What you are doing
Classify every column header of the aligned table into one of **three**
categories. The result drives how stage 7 splits the table into per-parameter
sub-tables.

> **Note — fewer categories than the summary pipeline.** Individual data has no
> statistic / variation / interval / P-value decomposition, and the parameter
> *type* is the header itself (handled in stage 8), so there is no
> `Parameter type` / `Parameter unit` / `P value` category here — only the three
> below.

## Inputs (read from the scratch directory)
- `05_param_aligned.md` — the aligned table from stage 5 (one subject per row,
  parameter types as headers).

Read this file now.

## Categories
Assign each column header exactly one of:
- **`Patient ID`** — a column that identifies the individual subject (an
  explicit patient/subject/case id, or an inferred unique unit). **At least one
  column must be `Patient ID`.** A column that is *only* a generic subject
  number with no identifying role is `Uncategorized`.
- **`Parameter value`** — a column holding an individual subject's measured PK
  value for some parameter (the header names the parameter, e.g.
  `Cmax (ng/mL)`, `Infant's plasma (ng/ml)`; the cells are that subject's value).
- **`Uncategorized`** — anything else (dose, other drugs, phenotype, outcomes,
  bleeding, …). These are carried along but not curated.

## Reasoning then answer
For each header, state the category and a one-clause reason, then produce the
mapping. Make sure at least one header is `Patient ID`.

## Output of this stage
A JSON object mapping each header to its category:

```json
{"categorized_headers": {"<header_1>": "<category_1>", "<header_2>": "<category_2>"}}
```

Before continuing, sanity-check:
- every header in `05_param_aligned.md` appears as a key exactly once,
- every value is one of the three categories above,
- at least one header is `Patient ID`,
- at least one header is `Parameter value`.

If that check fails, redo this stage once.

Then **write the JSON to `06_header_categories.md`** in the scratch directory.
Stage 7 reads that file.

## Worked example

**`05_param_aligned.md`** has headers `ID`, `Dose (mg/d)`, `Mother's plasma
(ng/ml)`, `Infant's plasma (ng/ml)`.

**Reasoning**: `ID` identifies the subject → `Patient ID`. `Dose (mg/d)` is not
a curated PK value → `Uncategorized`. The two plasma columns hold each subject's
measured concentration → `Parameter value`.

**Result** (`06_header_categories.md`):

```json
{"categorized_headers": {"ID": "Patient ID", "Dose (mg/d)": "Uncategorized", "Mother's plasma (ng/ml)": "Parameter value", "Infant's plasma (ng/ml)": "Parameter value"}}
```
