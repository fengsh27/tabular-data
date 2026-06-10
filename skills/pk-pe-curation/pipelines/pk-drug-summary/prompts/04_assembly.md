# Stage 04 — Assembly

This file is loaded by `procedure.md`. It joins the three row-aligned stage tables
into the final 11-column dataset.

## Inputs (read from the scratch directory)
- `03_drug_refined.md` — `[Drug/Metabolite name, Dose amount, Dose unit, Dose
  frequency, Dose schedule, Dose route]` (Stage 3).
- `02_patient_refined.md` — `[Population, Pregnancy stage, Pediatric/Gestational
  age, Population N]` (Stage 2).
- `01_drug_info.md` — used **only** for its `Source text` column (Stage 1).

Read these now.

## How to assemble
All three tables describe the **same rows in the same order** (stages 2 and 3 are
column-wise refinements of stage 1). So this is a **positional horizontal join**,
row *i* of each table forming output row *i*:

1. Confirm all three tables have the **same row count**. If they do not, do not
   guess an alignment — note the mismatch, align as many rows as the carried
   `Population N` key allows, and report the rest as unresolved.
2. Build each output row, in this exact column order:
   `Drug/Metabolite name, Dose amount, Dose unit, Dose frequency, Dose schedule,
   Dose route` (from Stage 3) → `Population, Pregnancy stage, Pediatric/Gestational
   age, Population N` (from Stage 2) → `Note` (the Stage-1 `Source text`,
   **renamed** to `Note`).
3. Drop a row only if it is fully empty / all-`N/A` across every column.

That yields the 11 columns:
`Drug/Metabolite name, Dose amount, Dose unit, Dose frequency, Dose schedule,
Dose route, Population, Pregnancy stage, Pediatric/Gestational age, Population N,
Note`.

## Output of this stage
Write the assembled 11-column table to `04_final.csv` in the scratch directory
(CSV with the header row exactly as above). Stage 5 verifies and corrects it in
place.

Before continuing, sanity-check:
- exactly 11 columns in the schema order,
- row count matches the stage tables (modulo any all-empty rows dropped),
- `Note` came from Stage 1's `Source text`.
