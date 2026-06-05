# Stage 10 — Patient matching

This file is loaded by `SKILL.md` during the PK individual curation procedure.
**Only run this prompt if the shortcut in SKILL.md does not apply** (i.e. the
refined patient table has more than one row). With a single cohort, assign it to
every row directly without reasoning.

## What you are doing
Attach the full cohort — a row of the refined patient table — to each sub-table
row, by looking up that row's **`Patient ID`**. Because each sub-table row
already carries its `Patient ID`, this is essentially a deterministic **lookup
by Patient ID**, not a fuzzy match.

## Inputs (read from the scratch directory)
- `07_subtables.md` — the per-parameter sub-tables (each row has `Row`,
  `Patient ID`, …) — the rows to label.
- `03_patient_refined.md` — the `[Patient ID, Population, Pregnancy stage,
  Pediatric/Gestational age]` cohorts.

Read these files now.

## Procedure (per sub-table)
1. Process **every row** of the sub-table, keeping each row's `Row` join-key
   value — output exactly the same `Row` values as the sub-table.
2. For each row, take its `Patient ID` and find the **matching** row in
   `03_patient_refined.md` (same Patient ID). Attach that cohort's `Population`,
   `Pregnancy stage`, and `Pediatric/Gestational age`.
3. If a Patient ID has no entry in the refined patient table (it should — both
   come from the same subjects), set those three fields to `N/A` and note it.

## Reasoning then answer
For each row, state which Patient ID you looked up and the cohort you attached,
then produce the result.

## Output of this stage
For each sub-table, a five-column markdown table — the `Row` join key plus the
matched cohort (Patient ID included so assembly can carry it):

```
| Row | Patient ID | Population | Pregnancy stage | Pediatric/Gestational age |
| --- | --- | --- | --- | --- |
| 1 | 1 | Maternal | Trimester 3 | N/A |
| 2 | 3 | Maternal | Trimester 3 | N/A |
```

Use the same `## Sub-table N` headings as `07_subtables.md`.

Before continuing, sanity-check:
- each output table has the **same `Row` values** as its sub-table (same set,
  same order),
- each row's `Patient ID` is unchanged from the sub-table,
- every attached cohort appears in `03_patient_refined.md` (or the three demo
  fields are `N/A`).

If that check fails, redo this stage once.

Then **write the result to `10_patient_matched.md`** in the scratch directory.
Stage 12 (assembly) reads that file.

## Note on the shortcut
If `03_patient_refined.md` has a **single** cohort, the SKILL.md shortcut
applies: attach that one cohort (with each row's own `Patient ID`) to every row
without running this prompt.
