# Stage 04 — Assembly

This file is loaded by `procedure.md`. It joins the three row-aligned stage tables
and applies the final column renames to produce the 15-column dataset.

## Inputs (read from the scratch directory)
- `01_characteristic_info.md` — used for `Population characteristic`,
  `Characteristic sub-category`, `Source text` (Stage 1).
- `02_patient_refined.md` — `[Population, Pregnancy stage, Pediatric/Gestational
  age, Population N]` (Stage 2).
- `03_characteristic_refined.md` — `[Main value, Unit, Statistics type, Variation
  type, Variation value, Interval type, Lower bound, Upper bound]` (Stage 3).

Read these now.

## How to assemble
All three tables describe the **same rows in the same order**. This is a
**positional horizontal join**, row *i* of each forming output row *i*:

1. Confirm all three have the **same row count**. If not, do not guess — note the
   mismatch, align as many rows as the carried `Population N` allows, and report
   the rest as unresolved.
2. Build each output row, taking columns from the stage tables and **renaming**
   them to the final schema names, in this exact order:

   | Final column | Source |
   |--------------|--------|
   | Characteristic | Stage 1 `Population characteristic` |
   | Characteristic subcategory | Stage 1 `Characteristic sub-category` |
   | Characteristic unit | Stage 3 `Unit` |
   | Characteristic value | Stage 3 `Main value` |
   | Statistics type | Stage 3 `Statistics type` |
   | Variation type | Stage 3 `Variation type` |
   | Variation value | Stage 3 `Variation value` |
   | Interval type | Stage 3 `Interval type` |
   | Lower bound | Stage 3 `Lower bound` |
   | Upper bound | Stage 3 `Upper bound` |
   | Population | Stage 2 `Population` |
   | Pregnancy stage | Stage 2 `Pregnancy stage` |
   | Pediatric/Gestational age | Stage 2 `Pediatric/Gestational age` |
   | Subject N | Stage 2 `Population N` |
   | Note | Stage 1 `Source text` |

That yields the 15 columns:
`Characteristic, Characteristic subcategory, Characteristic unit, Characteristic
value, Statistics type, Variation type, Variation value, Interval type, Lower
bound, Upper bound, Population, Pregnancy stage, Pediatric/Gestational age,
Subject N, Note`.

## Output of this stage
Write the assembled 15-column table to `04_final.csv` (CSV with the header row
exactly as above). Stage 5 verifies and corrects it in place.

Before continuing, sanity-check:
- exactly 15 columns in the schema order,
- `Population characteristic`→`Characteristic`, `Population N`→`Subject N`,
  `Source text`→`Note`, `Unit`→`Characteristic unit`, `Main value`→`Characteristic
  value`, `Characteristic sub-category`→`Characteristic subcategory` were applied,
- row count equals the stage tables'.
