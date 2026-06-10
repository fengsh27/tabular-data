# Stage 04 — Assembly

This file is loaded by `procedure.md`. It joins the three row-aligned stage tables
and applies the final column renames to produce the 9-column dataset (pre-cleanup).

## Inputs (read from the scratch directory)
- `01_characteristic_info.md` — used for `Patient ID`, `Patient characteristic`,
  `Characteristic sub-category`, `Source text` (Stage 1).
- `02_patient_refined.md` — `[Patient ID, Population, Pregnancy stage,
  Pediatric/Gestational age]` (Stage 2).
- `03_characteristic_refined.md` — `[Patient ID, Main value, Unit]` (Stage 3).

Read these now.

## How to assemble
All three tables describe the **same rows in the same order**. This is a
**positional horizontal join**, row *i* of each forming output row *i*:

1. Confirm all three have the **same row count**. If not, do not guess — note the
   mismatch, align as many rows as the carried `Patient ID` allows, and report the
   rest as unresolved.
2. Build each output row, taking columns from the stage tables and **renaming**
   them to the final schema names, in this exact order:

   | Final column | Source |
   |--------------|--------|
   | Patient ID | Stage 1 `Patient ID` |
   | Characteristic | Stage 1 `Patient characteristic` |
   | Characteristic subcategory | Stage 1 `Characteristic sub-category` |
   | Characteristic unit | Stage 3 `Unit` |
   | Characteristic value | Stage 3 `Main value` |
   | Population | Stage 2 `Population` |
   | Pregnancy stage | Stage 2 `Pregnancy stage` |
   | Pediatric/Gestational age | Stage 2 `Pediatric/Gestational age` |
   | Note | Stage 1 `Source text` |

That yields the 9 columns:
`Patient ID, Characteristic, Characteristic subcategory, Characteristic unit,
Characteristic value, Population, Pregnancy stage, Pediatric/Gestational age,
Note`.

Do **not** drop rows here — that is stage 5's job (the cleanup script).

## Output of this stage
Write the assembled 9-column table to `04_assembled.csv` (CSV with the header row
exactly as above). Stage 5 cleans it.

Before continuing, sanity-check:
- exactly 9 columns in the schema order, `Patient ID` first,
- `Patient characteristic`→`Characteristic`, `Characteristic sub-category`→
  `Characteristic subcategory`, `Unit`→`Characteristic unit`, `Main value`→
  `Characteristic value`, `Source text`→`Note` were applied,
- row count equals the stage tables'.
