# Stage 12 — Assembly

This file is loaded by `SKILL.md` as the assembly stage of the PK individual
curation procedure (it is followed by stage 13 row cleanup and stage 14
verification).

## What you are doing
Join the per-stage tables **on the `Row` key** (within each sub-table) into the
12-column individual schema. Mechanical join, not extraction — joining by key,
not by position, so a stage that reordered its rows cannot mis-attribute.

## Inputs (read from the scratch directory), per sub-table
Each carries the `Row` join key as its first column:
- `08_type_unit_value.md` — `[Row, Parameter type, Parameter unit, Parameter value]`
- `10_patient_matched.md` — `[Row, Patient ID, Population, Pregnancy stage, Pediatric/Gestational age]`
- `11_time.md` — `[Row, Time value, Time unit]`

Drug may be **`Row`-keyed** (if matching ran) or a **single broadcast row** (if
the stage-9 shortcut applied):
- `09_drug_matched.md` — `[Row, Drug name, Analyte, Specimen]`, or a single
  `[Drug name, Analyte, Specimen]` to broadcast to every row.

Read these files now.

## Procedure
1. For each `## Sub-table N`, take the `Row` keys from `08_type_unit_value.md` as
   the authoritative row set. Confirm `10_patient_matched.md` and `11_time.md`
   (and a `Row`-keyed `09_drug_matched.md`) have the **exact same set of `Row`
   values**. On any mismatch — missing / extra / duplicate `Row` — **do not
   guess**: report it and stop (a prior stage broke the join key).
2. For each `Row` key, gather that row's fields from each input **by matching the
   `Row` value**. For a broadcast drug table (single row, no `Row`), use that one
   combination for every row.
3. Assemble each row into the final **12 columns, in this order**:
   ```
   Patient ID, Drug name, Analyte, Specimen,
   Population, Pregnancy stage, Pediatric/Gestational age,
   Parameter type, Parameter unit, Parameter value,
   Time value, Time unit
   ```
   (`Patient ID` comes from `10_patient_matched.md`.) **Drop the `Row` helper
   column** — it is a join key, not output.
4. Stack all sub-tables' rows into one table.

## Output of this stage
Write the assembled table as CSV to `12_assembled.csv` (header row + one row per
measurement). Row-level cleanup (dropping N/A-value rows, time normalization,
dedupe) happens next in stage 13 — do **not** do it here.

Before finishing, sanity-check:
- column order matches the 12-column list above exactly,
- the `Row` helper column does **not** appear,
- row count equals the sum of the per-sub-table row counts.
