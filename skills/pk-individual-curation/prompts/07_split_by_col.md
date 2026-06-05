# Stage 07 — Split by columns

This file is loaded by `SKILL.md` during the PK individual curation procedure.

## What you are doing
Split the aligned table into one **sub-table per `Parameter value` column**.
Each sub-table is the long-format view of one parameter: one row per subject,
carrying that subject's `Patient ID`, the parameter name (taken from the
column header), and that subject's value.

> **Note — differs from the summary pipeline.** Here the parameter *type* is the
> value column's **header**, which becomes a `Parameter type` column broadcast
> down every row of the sub-table. (In pk-summary the parameter type was already
> a column; here we manufacture it from the header.)

## Inputs (read from the scratch directory)
- `05_param_aligned.md` — the aligned table (one subject per row).
- `06_header_categories.md` — the JSON header→category mapping from stage 6.

Read these files now.

## Splitting rule
For **each** header categorized `Parameter value` in `06_header_categories.md`,
build one sub-table with these columns:
- **`Row`** — a 1-based index, unique within the sub-table (the join key; see
  below).
- **`Patient ID`** — copied from the row's `Patient ID` column.
- **`Parameter type`** — the value column's **header**, repeated on every row.
- **`Parameter value`** — that subject's cell from the value column.

Produce one sub-table per `Parameter value` column, in column order. Keep every
subject row (one row per subject per parameter). `Uncategorized` columns are not
carried into the sub-tables (they were never curated); `Patient ID` is.

## The `Row` join key
Give every sub-table a leading `Row` column numbered `1, 2, 3, …`, unique within
that sub-table. Stages 08–11 carry this value through unchanged so stage 12 can
join them by `(Sub-table N, Row)` rather than by position. Keep it stable.

## Reasoning then answer
State how many `Parameter value` columns there are (= number of sub-tables) and
which header each sub-table came from, then produce them.

## Output of this stage
Write each sub-table under a heading `## Sub-table N` (N starting at 1):

```
## Sub-table 1

| Row | Patient ID | Parameter type | Parameter value |
| --- | --- | --- | --- |
| 1 | 1 | <value-column header> | <subject 1's value> |
| 2 | 2 | <value-column header> | <subject 2's value> |
```

Before continuing, sanity-check:
- one sub-table per `Parameter value` column,
- each sub-table's columns are exactly `Row, Patient ID, Parameter type, Parameter value`,
- `Parameter type` is the same header on every row of a sub-table,
- `Row` is 1..k with no gaps or repeats.

If that check fails, redo this stage once.

Then **write the sub-tables to `07_subtables.md`** in the scratch directory.
Stages 8–11 read that file.

## Worked example

`06_header_categories.md` marks two `Parameter value` columns — `Mother's plasma
(ng/ml)` and `Infant's plasma (ng/ml)` — and `ID` as `Patient ID`. → two
sub-tables:

```
## Sub-table 1

| Row | Patient ID | Parameter type | Parameter value |
| --- | --- | --- | --- |
| 1 | 1 | Mother's plasma (ng/ml) | 19.5 |
| 2 | 3 | Mother's plasma (ng/ml) | 14.4 |

## Sub-table 2

| Row | Patient ID | Parameter type | Parameter value |
| --- | --- | --- | --- |
| 1 | 1 | Infant's plasma (ng/ml) | NA |
| 2 | 3 | Infant's plasma (ng/ml) | 9 |
```
