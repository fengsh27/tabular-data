# Stage 13 — Assembly + row cleanup

This file is loaded by `SKILL.md` as the assembly stage of the PK summary
curation procedure (it is followed by stage 14, verification + correction).

## What you are doing
Join the per-stage tables **column-wise**, row for row, into the final curated
table; drop fully-empty rows; and rename two internal columns. This stage is
mechanical — a deterministic join, not an extraction. Every per-stage table has
the same number of rows, in the same order (one row per parameter measurement),
so they align positionally.

## Inputs (read from the scratch directory), per sub-table
- `09_drug_matched.md` — `[Drug name, Analyte, Specimen]`
  (or, if the stage-9 shortcut applied, the single drug combination broadcast
  to every row)
- `10_patient_matched.md` — `[Population, Pregnancy stage, Pediatric/Gestational
  age, Subject N]` (or the single cohort broadcast, per the stage-10 shortcut)
- `08_type_unit.md` — `[Parameter type, Parameter unit]`
- `11_param_values.md` — `[Main value, Statistics type, Variation type,
  Variation value, Interval type, Lower bound, Upper bound, P value]`
- `12_time.md` — `[Time value, Time unit]`

Read these files now.

## Procedure
1. For each `## Sub-table N`, confirm all five inputs have the **same row
   count**. If they disagree, do not guess — report the mismatch to the user
   and stop (a prior stage broke row alignment).
2. Concatenate the columns left-to-right in this order:
   `drug → patient → type/unit → values → time`.
3. Stack all sub-tables' rows into one table.
4. **Drop fully-empty rows** (rows that are entirely `N/A`/blank).
5. **Rename** two columns:
   - `Main value` → `Parameter value`
   - `Statistics type` → `Parameter statistic`

## Final columns (in order)
```
Drug name, Analyte, Specimen,
Population, Pregnancy stage, Pediatric/Gestational age, Subject N,
Parameter type, Parameter unit,
Parameter value, Parameter statistic, Variation type, Variation value,
Interval type, Lower bound, Upper bound, P value,
Time value, Time unit
```

## Output of this stage
Write the final table as CSV to `13_final.csv` (header row + one row per
measurement). Also present it to the user as a markdown table.

Before finishing, sanity-check:
- column order matches the list above exactly,
- row count equals the per-sub-table row counts minus any all-empty rows
  removed,
- the `Main value` / `Statistics type` headers no longer appear (they were
  renamed).

Then run the validation and verification passes described in `SKILL.md`.
