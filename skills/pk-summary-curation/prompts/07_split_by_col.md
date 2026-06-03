# Stage 07 — Split by columns

This file is loaded by `SKILL.md` during the PK summary curation procedure.

## What you are doing
Split the aligned table into one or more **sub-tables**, each describing a
single parameter type. Downstream stages (value extraction, matching) operate
on one sub-table at a time.

## Inputs (read from the scratch directory)
- `05_param_aligned.md` — the aligned table from stage 5.
- `06_header_categories.md` — the JSON header→category mapping from stage 6.

Read these files now.

## Splitting rule
Group the columns so that **each sub-table contains exactly one `Parameter
type` column and at most one `P value` column.** Carry the `Parameter type`
column (and any shared `Uncategorized`/identifier columns) into every
sub-table; distribute the `Parameter value` / `P value` columns across the
groups.

- If there is only **one** `Parameter type` column and **at most one** `P
  value` column, the table does **not** need splitting — the single sub-table
  is the whole table. (This is the common case.)
- Split only when there are multiple `Parameter type` columns and/or multiple
  `P value` columns, so that each resulting sub-table has exactly one parameter
  type and at most one P value.

## Reasoning then answer
State how many sub-tables result and which columns go into each, then produce
them.

## Output of this stage
Write each sub-table as a markdown table, separated by a heading line
`## Sub-table N` (N starting at 1). For the common single-sub-table case, emit
just `## Sub-table 1` followed by the whole table.

Before continuing, sanity-check:
- each sub-table has exactly one `Parameter type` column,
- each sub-table has at most one `P value` column,
- no `Parameter value` column was dropped or duplicated across sub-tables.

If that check fails, redo this stage once.

Then **write the sub-tables to `07_subtables.md`** in the scratch directory.
Stages 8–12 read that file.

## Worked example (single sub-table — no split needed)

**`05_param_aligned.md`** has one `Parameter type` column and one
`Parameter value` column (`Mean CI 95%`) → no split.

**Result** (`07_subtables.md`):

```
## Sub-table 1

| Parameter type | Mean CI 95% |
| --- | --- |
| Cord blood (ng/ml) | 6.78 (5.39–8.17) |
| Maternal blood (ng/ml) | 9.91 (7.68–12.14) |
| Collection time(min) | 293.4 (163.2–423) |
| Cord blood/maternal blood | 0.73 (0.52–0.94) |
```
