# Stage 02 — Patient / population refine

This file is loaded by `procedure.md`. It is a **thin wrapper** over the shared
demographic-refinement procedure in
`refine_population.md`. Read that file and follow it, with
the parameters below filled in for the PK drug individual schema.

## Parameters for the shared procedure
- **`<INPUT_TABLE>`** = the stage-1 table `01_drug_info.md` (read it from the
  scratch directory).
- **`<KEY-COLUMN>`** = `Patient ID` — carry it through unchanged; it ties each
  refined row back to its stage-1 patient row.
- **`<OUTPUT-COLUMNS>`** = `Patient ID, Population, Pregnancy stage, Pediatric/Gestational age`
  (the key column **first**).
- **`<TITLE+FULLTEXT>`** = `inputs.md` (title + full text). The stage-1 table has
  **no Population column** — infer each patient's `Population`, `Pregnancy stage`,
  and `Pediatric/Gestational age` from the full text, matched to that `Patient ID`.

## Reminder of the per-row rule
One output row per `01_drug_info.md` row, in the **same order**, with the same
`Patient ID`. Map `Population` to a canonical category (`Nonpregnant` /
`Maternal` / `Pediatric` / `Adults`) and `Pregnancy stage` to its canonical
category when they fit; keep original wording otherwise. Fill
`Pediatric/Gestational age` only when the age is **explicitly stated** (never
inferred from a measurement or dosing time).

## Output of this stage
A markdown table with columns exactly
`Patient ID, Population, Pregnancy stage, Pediatric/Gestational age`, one row per
stage-1 row. **Write it to `02_patient_refined.md`** in the scratch directory.
Stage 4 reads it.
