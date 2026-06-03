# Stage 10 — Patient matching

This file is loaded by `SKILL.md` during the PK summary curation procedure.
**Only run this prompt if the shortcut in SKILL.md does not apply** (i.e. the
refined patient table has more than one row). With a single cohort, assign it
to every row directly without reasoning.

## What you are doing
Attach one cohort — a row of the refined patient table — to each value-bearing
row of the sub-table.

## Inputs (read from the scratch directory)
- `07_subtables.md` — the per-parameter sub-tables (the rows to label).
- `03_patient_refined.md` — the `[Population, Pregnancy stage, Pediatric/
  Gestational age, Subject N]` cohorts.
- `05_param_aligned.md` — the aligned main table, for context.
- `inputs.md` — caption + title.

Read these files now.

## Procedure (per sub-table)
1. Process **every row** of the sub-table, in order — output exactly as many
   entries as the sub-table has rows.
2. For each row, find the **best-matching** cohort in `03_patient_refined.md`:
   - First find the corresponding main-table row (by Parameter value / P value).
   - Use its **Subject N** to pick the matching cohort row.
3. **Subject N may vary across parameters within one population** because of
   data availability — match each row to the correct N from context. (E.g. a
   group with total N=10 but a parameter measured in only 9 → match the N=9
   cohort row for that parameter.)
4. If no match exists after applying all criteria, assign `N/A` (last resort).

## Reasoning then answer
For each row, state which cohort you matched and the Subject N you used, then
produce the result.

## Output of this stage
For each sub-table, a four-column markdown table — one row per sub-table row,
in order:

```
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| <matched cohort row> | … | … | … |
```

Use the same `## Sub-table N` headings as `07_subtables.md`.

Before continuing, sanity-check:
- each output table has the **same number of rows** as its sub-table, in order,
- every cohort row appears in `03_patient_refined.md` (or is `N/A`).

If that check fails, redo this stage once.

Then **write the result to `10_patient_matched.md`** in the scratch directory.
Stage 13 (assembly) reads that file.

## Note on the shortcut
For `16143486_table_4` the refined patient table has a **single** cohort
(`Maternal | Delivery | N/A | 8`), so the SKILL.md shortcut applies: assign
that cohort to every row without running this matching prompt.
