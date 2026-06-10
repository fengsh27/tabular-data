# Stage 02 — Patient / population refine

This file is loaded by `procedure.md`. It is a **thin wrapper** over the shared
demographic-refinement procedure in
`curation-common/refine_population.md`. Read that file and follow it, with
the parameters below filled in for the PK population summary schema.

## Parameters for the shared procedure
- **`<INPUT_TABLE>`** = the stage-1 table `01_characteristic_info.md` (read it
  from the scratch directory). Its `Population` / `Population N` columns are the
  inputs to refine; the other stage-1 columns are ignored here.
- **`<KEY-COLUMN>`** = `Population N` — carry it through unchanged.
- **`<OUTPUT-COLUMNS>`** = `Population, Pregnancy stage, Pediatric/Gestational age, Population N`
  (the key column **last**).
- **`<TITLE+FULLTEXT>`** = `inputs.md` (title + full text).

## Reminder of the per-row rule
One output row per `01_characteristic_info.md` row, in the **same order**, with the
same `Population N`. Map `Population` to a canonical category (`Nonpregnant` /
`Maternal` / `Pediatric` / `Adults`) and `Pregnancy stage` to its canonical
category when they fit; keep original wording otherwise. Fill
`Pediatric/Gestational age` only when the age is **explicitly stated**.

## Output of this stage
A markdown table with columns exactly
`Population, Pregnancy stage, Pediatric/Gestational age, Population N`, one row per
stage-1 row. **Write it to `02_patient_refined.md`**. Stage 4 reads it.
