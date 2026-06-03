# Stage 06 — Header categorization

This file is loaded by `SKILL.md` during the PK summary curation procedure.

## What you are doing
Classify every column header of the aligned table into one of five categories.
The result drives how stage 7 splits the table into per-parameter sub-tables.

## Inputs (read from the scratch directory)
- `05_param_aligned.md` — the aligned table from stage 5 (its first column is
  `Parameter type`).

Read this file now; do not rely on the table text remaining in the
conversation, which may have been summarized.

## Categories
Assign each column header exactly one of:
- **`Parameter type`** — describes the type of PK parameter (the column literally
  named `Parameter type`, and any other column that names parameters).
- **`Parameter unit`** — a column that contains **only** a unit. Note: a header
  like `fentanyl (ng/ml)` is *not* a unit column — it carries a parameter name
  too, so it is `Parameter type`.
- **`Parameter value`** — columns holding numeric parameter values (e.g.
  `Mean ± s.d.`, `Median`, `Range`, or a value column named after a cohort).
- **`P value`** — columns holding statistical P values.
- **`Uncategorized`** — anything else. A column that is only a subject count
  (`N`, `n`) is `Uncategorized`.

## Reasoning then answer
For each header, state the category and a one-clause reason, then produce the
mapping.

## Output of this stage
A JSON object mapping each header to its category:

```json
{"categorized_headers": {"<header_1>": "<category_1>", "<header_2>": "<category_2>"}}
```

Before continuing, sanity-check:
- every header in `05_param_aligned.md` appears as a key exactly once,
- every value is one of the five categories above,
- at least one header is categorized as `Parameter type`.

If that check fails, redo this stage once.

Then **write the JSON to `06_header_categories.md`** in the scratch directory.
Stage 7 reads that file, not this message.

## Worked example

**`05_param_aligned.md`**:

```
| Parameter type | Mean CI 95% |
| --- | --- |
| Cord blood (ng/ml) | 6.78 (5.39–8.17) |
| … | … |
```

**Reasoning**: `Parameter type` names the parameters → `Parameter type`. The
`Mean CI 95%` column holds the numeric values → `Parameter value`.

**Result** (`06_header_categories.md`):

```json
{"categorized_headers": {"Parameter type": "Parameter type", "Mean CI 95%": "Parameter value"}}
```
