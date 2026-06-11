# Stage 08 — Parameter type + unit + value extraction

This file is loaded by `procedure.md` during the PK individual curation procedure.

## What you are doing
For each row of each sub-table, turn the raw `Parameter type` (a column header
carried down in stage 7) into a clean **refined Parameter type**, its
**Parameter unit**, and the individual **Parameter value** — three aligned
fields per row.

> **Note — differs from the summary pipeline.** Individual data has no statistic
> decomposition: each cell is one subject's single value, so this stage emits
> the value directly (there is no separate value-extraction stage downstream).

## Inputs (read from the scratch directory)
- `07_subtables.md` — the per-parameter sub-tables `[Row, Patient ID,
  Parameter type, Parameter value]`.
- `05_param_aligned.md` — the aligned main table, for context.
- `inputs.md` — caption + footnotes + title, for refining ambiguous
  names/units.

Read these files now.

## Procedure (per sub-table, per row)
Keep each row's `Row` join-key value, and produce three fields:
1. **Parameter type** — refine the raw header into a clear PK parameter name,
   **without losing meaningful tokens** (biological matrix, subject/context like
   maternal/fetal/cord, timing like trough/peak). Examples:
   - `Mother's PL III trimester (ng/ml)` → `Maternal plasma concentration - third trimester`
   - `Infant's PL (ng/ml)` → `Infant plasma concentration`
   - `Cmax (ng/mL)` → `Cmax`
   Keep the core concept clear but do **not** strip qualifiers down to a bare
   `Concentration`.
2. **Parameter unit** — the unit, preferring what is explicit in the sub-table
   header; clarify from the main table/caption only if needed (e.g. `ng/ml`;
   a dimensionless ratio → `unitless`).
3. **Parameter value** — the individual value, **copied verbatim** from the
   sub-table's `Parameter value` cell. No arithmetic, no inference.

If any of the three cannot be confidently produced for a row, use `N/A` for all
three of that row.

## Reasoning then answer
Show, per row, how you derived the refined type and unit and which value you
copied, then produce the table.

## Output of this stage
For each sub-table, a four-column markdown table — the `Row` join key plus the
three fields. Use the same `## Sub-table N` headings as `07_subtables.md`:

```
| Row | Parameter type | Parameter unit | Parameter value |
| --- | --- | --- | --- |
| 1 | Maternal plasma concentration - third trimester | ng/ml | 19.5 |
| 2 | Maternal plasma concentration - third trimester | ng/ml | 14.4 |
```

Before continuing, sanity-check:
- each output table has the **same `Row` values** as its sub-table (same set,
  same order),
- `Parameter value` is copied verbatim from the sub-table (no calculation),
- columns are exactly `Row, Parameter type, Parameter unit, Parameter value`.

If that check fails, redo this stage once.

Then **write the result to `08_type_unit_value.md`** in the scratch directory.
Stage 12 (assembly) reads that file.
