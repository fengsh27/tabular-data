# Stage 04 — Individual-data deletion

This file is loaded by `SKILL.md` during the PK summary curation procedure.

## What you are doing
Remove rows (and, if needed, columns) that report **individual-subject**
results, keeping only **summary / aggregate / group-level** data. This skill
curates summary statistics; per-subject rows belong to the pk-individual
pipeline, not here.

## Inputs (read from the scratch directory)
- `00_markdown_table.md` — the source PK table in markdown (Stage 0 output).

Read this file now; do not rely on the table text remaining in the
conversation, which may have been summarized.

## What to remove vs keep
- **Remove** rows that describe a specific individual: per-patient rows,
  per-subject rows, rows keyed by a Patient ID / Subject number / Volunteer
  number, and any personally identifiable data.
- **Keep** summary statistics and group-level information: rows reporting a
  Mean, Median, Geometric mean, Range, SD, CI, or an `N=` / `n=` count. These
  are aggregate, not individual-specific — never drop them.
- **Keep all columns by default.** Only drop a column if it contains nothing
  but individual-level values once the individual rows are gone. (A column like
  "Parturient" or "Volunteer" is usually retained — after deletion its cell on
  the summary row holds the aggregate label, e.g. "Mean CI 95%".)

## Procedure
1. Read the table and identify which rows are individual-level and which are
   summary/aggregate.
2. If **every** row is already summary-level (the table has no individual rows
   at all), the table is fine as-is — pass it through unchanged.
3. Otherwise, drop the individual rows, keep the summary rows and (by default)
   all columns.

## Reasoning then answer
State which rows you classified as individual-level and which as summary, and
why, before producing the filtered table.

## Output of this stage
Produce the filtered markdown table — same columns and header, only the kept
rows. Preserve the original column order and cell contents exactly (do not
re-format numbers or intervals).

Before continuing, sanity-check:
- at least one row remains (a table with no summary row at all should pass
  through unchanged rather than become empty — if you would otherwise empty it,
  keep the original and note this to the user),
- no individual-subject row survived,
- the header row is unchanged.

If that check fails, redo this stage once.

Then **write the result to `04_summary_only.md`** in the scratch directory.
Stage 5 (parameter-type alignment) reads that file, not this message — so write
the complete, exact table.

## Worked example

**`00_markdown_table.md`** (transplacental lorazepam, n = 8):

```
| Parturient | Cord blood (ng/ml) | Maternal blood (ng/ml) | Collection time(min) | Cord blood/maternal blood |
| --- | --- | --- | --- | --- |
| 1 | 5.77 | 14.74 | 135 | 0.392 |
| 2 | 6.82 | 7.95 | 426 | 0.858 |
| … | … | … | … | … |
| 8 | 9.45 | 10.35 | 207 | 0.913 |
| Mean CI 95% | 6.78 (5.39–8.17) | 9.91 (7.68–12.14) | 293.4 (163.2–423) | 0.73 (0.52–0.94) |
```

**Reasoning**: rows keyed by Parturient 1–8 are individual-subject results →
remove. The "Mean CI 95%" row is a group-level summary → keep. All columns are
retained; the Parturient column's summary-row cell holds the aggregate label.

**Result** (`04_summary_only.md`):

```
| Parturient | Cord blood (ng/ml) | Maternal blood (ng/ml) | Collection time(min) | Cord blood/maternal blood |
| --- | --- | --- | --- | --- |
| Mean CI 95% | 6.78 (5.39–8.17) | 9.91 (7.68–12.14) | 293.4 (163.2–423) | 0.73 (0.52–0.94) |
```
