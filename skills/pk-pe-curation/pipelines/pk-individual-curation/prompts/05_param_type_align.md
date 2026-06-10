# Stage 05 — Parameter-type alignment

This file is loaded by `procedure.md` during the PK individual curation procedure.

## What you are doing
Make sure the table is oriented so that **each row is one subject** and the PK
**parameter types are the column headers**. Individual-data tables come both
ways; this stage normalizes the orientation.

> **Note — opposite of the summary pipeline.** For *individual* data the goal
> orientation is rows = subjects, parameters = column headers. So the transpose
> condition is the **reverse** of pk-summary: here you transpose only when the
> parameter types are sitting in a row-header column, not when they are headers.

## Inputs (read from the scratch directory)
- `04_individual_only.md` — the individual-only table from stage 4 (per-subject
  rows, with the aggregate/summary rows removed).

Read this file now; do not rely on the table text remaining in the
conversation, which may have been summarized.

## Decide the table's orientation
Look at how the PK parameter type (Cmax, AUC, t½, a measured concentration,
etc.) is expressed:

- **Case A — parameters are the column headers** (e.g. headers like
  `Cmax (ng/mL)`, `Mother's plasma (ng/ml)`, `Infant's plasma (ng/ml)`), and
  each data row is one subject. → **Keep the table as-is.** This is the common
  case for individual data. Do **not** transpose.

- **Case B — parameters are listed down a single row-header column** (one column
  whose cells are `Cmax`, `AUC`, …, and the other columns are the subjects). →
  **Transpose** the table so that each subject becomes a column-group and the
  parameter types become the headers — i.e. flip it into the Case-A shape.

If unsure, ask: "is there one column whose cells name the parameters?" If yes →
Case B (transpose). If the parameters are spread across the headers and each row
is a subject → Case A (keep).

## Reasoning then answer
State which case applies and why (point to the subject rows and the parameter
headers, or to the single column that names the parameters), then produce the
aligned table.

## Output of this stage
A markdown table with **one subject per row** and **parameter types as column
headers** (plus a patient-identifier column and any extra columns). Do not
reformat the values; only change orientation if Case B.

Before continuing, sanity-check:
- each data row corresponds to a single subject,
- the parameter types appear as column headers (not down a column),
- no value cells were lost, merged, or reformatted.

If that check fails, redo this stage once.

Then **write the result to `05_param_aligned.md`** in the scratch directory.
Stages 6 and 7 read that file.

## Worked example (Case A — keep, the common case)

**`04_individual_only.md`** (each row is one patient; parameters are headers):

```
| ID | Cmax (ng/mL) | AUC (ng·h/mL) |
| --- | --- | --- |
| 1 | 56.1 | 822.5 |
| 2 | 42.2 | 601.5 |
| 3 | 29.3 | 253.3 |
```

**Reasoning**: each row is one subject (ID 1, 2, 3) and the parameters
(`Cmax`, `AUC`) are the column headers → Case A → keep as-is.

**Result** (`05_param_aligned.md`): identical to the input (unchanged).
