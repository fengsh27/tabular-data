# Stage 08 — Parameter type + unit extraction

This file is loaded by `SKILL.md` during the PK summary curation procedure.

## What you are doing
For each sub-table, turn its `Parameter type` column into a clean pair of
columns: a refined **Parameter type** and its **Parameter unit**. The raw
parameter labels are often coarse (e.g. a specimen-concentration header); use
the main table and caption to refine them into proper PK parameter names and
units.

## Inputs (read from the scratch directory)
- `07_subtables.md` — the per-parameter sub-tables from stage 7.
- `05_param_aligned.md` — the aligned main table, for context.
- `inputs.md` — caption + footnotes + title, for refining ambiguous names/units.

Read these files now.

## Procedure (per sub-table)
1. Take the `Parameter type` column of the sub-table, row by row.
2. For each row produce two values: a refined **Parameter type** and a
   **Parameter unit**, inferring from the main table/caption when the sub-table
   label is too coarse (e.g. `Cord blood (ng/ml)` → type "Cord blood
   concentration", unit "ng/ml"; `Collection time(min)` → "Sample collection
   time", "minutes"; a dimensionless ratio → unit "unitless").
3. **Process exactly the rows of the sub-table — same count, same order.** For a
   row you cannot extract, use `N/A` for both values.

## Reasoning then answer
Show, per row, how you derived the refined type and unit, then produce the
table.

## Output of this stage
For each sub-table, a two-column markdown table:

```
| Parameter type | Parameter unit |
| --- | --- |
| <refined type> | <unit> |
```

Use the same `## Sub-table N` headings as `07_subtables.md`, so type/unit
tables stay aligned with their sub-tables.

Before continuing, sanity-check:
- each output table has the **same number of rows** as its sub-table, in order,
- columns are exactly `Parameter type` and `Parameter unit`.

If that check fails, redo this stage once.

Then **write the result to `08_type_unit.md`** in the scratch directory.
Stages 11 and 13 read that file.

## Worked example

**`07_subtables.md` → Sub-table 1** `Parameter type` column:
`Cord blood (ng/ml)`, `Maternal blood (ng/ml)`, `Collection time(min)`,
`Cord blood/maternal blood`.

**Result** (`08_type_unit.md`):

```
## Sub-table 1

| Parameter type | Parameter unit |
| --- | --- |
| Cord blood concentration | ng/ml |
| Maternal blood concentration | ng/ml |
| Sample collection time | minutes |
| Cord blood to maternal blood ratio | unitless |
```
