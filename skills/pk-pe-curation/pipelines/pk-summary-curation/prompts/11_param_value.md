# Stage 11 — Parameter value extraction

This file is loaded by `procedure.md` during the PK summary curation procedure.

## What you are doing
For each row of each sub-table, decompose the reported value into a fixed set
of numeric/statistic fields. **No arithmetic** — every value is copied directly
from the source.

## Inputs (read from the scratch directory)
- `07_subtables.md` — the per-parameter sub-tables (one Parameter value /
  P value column each).
- `05_param_aligned.md` — the aligned main table, for context (and the P-value
  column, which must be read from the main table).
- `inputs.md` — caption + title.

Read these files now.

## Output columns (exactly these, in this order)
Start each row with the **`Row`** join key, then the eight value fields:

0. **Row** — copy the `Row` index of the sub-table row verbatim (see "The `Row`
   join key" below). This is how stage 13 re-aligns this table with the others;
   do not renumber or reorder.
1. **Main value** — the single main value (not a range).
2. **Statistics type** — how the main value was summarized. Use exactly one of
   the **canonical values**: `Mean`, `Median`, `Geometric mean`,
   `Arithmetic mean`, `Count`, or `N/A`. **Required — always fill this in**
   (use `N/A` only if the source truly gives no central value). Do **not** put
   an interval label (e.g. `Range`) here — a range belongs in `Interval type`
   with its `Lower bound` / `Upper bound`.
3. **Variation type** — the variability measure: `SD`, `CV%`, `SEM`, …
4. **Variation value** — the single value of that variation (not a range).
5. **Interval type** — `95% CI`, `Range`, `IQR`, …
6. **Lower bound** — lower end of the interval.
7. **Upper bound** — upper end of the interval.
8. **P value** — extracted directly from the main table's P-value column.

## The `Row` join key
Each sub-table in `07_subtables.md` has a leading `Row` column (an integer that
is unique within its sub-table). Every per-row stage (08–12) carries that same
`Row` value through unchanged, so stage 13 can join the stages **by `Row`**
instead of by position. Copy the `Row` value into each output row exactly; never
invent, renumber, or reorder it.

## Rules
- An interval of two numbers goes into **Lower bound / Upper bound**, never into
  Variation value.
- For any field that does not apply, enter `N/A`.
- **No calculations** — copy values verbatim from the source.
- Process **exactly the rows of the sub-table**, in order — same count. For a
  row you cannot extract, enter `N/A` across all eight fields.

## Reasoning then answer
Show, per row, how you split the reported value into the eight fields, then
produce the table.

## Output of this stage
For each sub-table, a markdown table with exactly the nine columns above
(`Row` + the eight value fields), in order. Use the same `## Sub-table N`
headings as `07_subtables.md`.

Before continuing, sanity-check:
- each output table has the **same number of rows** as its sub-table, with the
  **same `Row` values** (same set, same order),
- `Statistics type` is filled (never `N/A`) for any row that has a Main value,
- no interval was placed in Variation value.

If that check fails, redo this stage once.

Then **write the result to `11_param_values.md`** in the scratch directory.
Stage 13 (assembly) reads that file.

## Worked example

A sub-table row with `Row` = 1 whose value is `6.78 (5.39–8.17)`, summarized as
a mean with a 95% CI:

```
| Row | Main value | Statistics type | Variation type | Variation value | Interval type | Lower bound | Upper bound | P value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.78 | Mean | N/A | N/A | 95% CI | 5.39 | 8.17 | N/A |
```
