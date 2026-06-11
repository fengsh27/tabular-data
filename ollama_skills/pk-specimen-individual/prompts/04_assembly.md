# Stage 04 — Assembly

This file is loaded by `procedure.md`. It joins the three row-aligned stage tables
into the 9-column dataset (pre-cleanup).

## Inputs (read from the scratch directory)
- `01_specimen_info.md` — used for its `Patient ID`, `Specimen`, and `Sample N`
  columns (Stage 1).
- `02_patient_refined.md` — `[Patient ID, Population, Pregnancy stage,
  Pediatric/Gestational age]` (Stage 2).
- `03_time.md` — `[Sample time, Time unit, Source text]` (Stage 3).

Read these now.

## How to assemble
All three tables describe the **same rows in the same order** (stages 2 and 3 are
row-aligned refinements of stage 1). So this is a **positional horizontal join**,
row *i* of each forming output row *i*:

1. Confirm all three tables have the **same row count**. If not, do not guess —
   note the mismatch, align as many rows as the carried `Patient ID` allows, and
   report the rest as unresolved.
2. Build each output row in this exact column order:
   `Patient ID, Specimen, Sample N` (from Stage 1) → `Population, Pregnancy stage,
   Pediatric/Gestational age` (from Stage 2) → `Sample time, Time unit` (from
   Stage 3) → `Note` (the Stage-3 `Source text`, **renamed** to `Note`).

That yields the 9 columns:
`Patient ID, Specimen, Sample N, Population, Pregnancy stage, Pediatric/Gestational
age, Sample time, Time unit, Note`.

Do **not** drop or dedupe rows here — that is stage 5's job (the cleanup script).

## Output of this stage
Write the assembled 9-column table to `04_assembled.csv` in the scratch directory
(CSV with the header row exactly as above). Stage 5 cleans it.

Before continuing, sanity-check:
- exactly 9 columns in the schema order, `Patient ID` first,
- the last column is `Note` (from Stage 3's `Source text`),
- row count equals the stage tables'.
