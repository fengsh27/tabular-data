# Stage 13 — Assembly + row cleanup

This file is loaded by `SKILL.md` as the assembly stage of the PK summary
curation procedure (it is followed by stage 14, verification + correction).

## What you are doing
Join the per-stage tables into the final curated table; drop fully-empty rows;
and rename two internal columns. This stage is mechanical — a deterministic
join, not an extraction. The per-stage tables are joined **on the `Row` key**
(within each sub-table), not by position, so a stage that accidentally
reordered its rows cannot silently mis-attribute a value to the wrong parameter.

## Inputs (read from the scratch directory), per sub-table
Each of these carries the `Row` join key as its first column:
- `08_type_unit.md` — `[Row, Parameter type, Parameter unit]`
- `11_param_values.md` — `[Row, Main value, Statistics type, Variation type,
  Variation value, Interval type, Lower bound, Upper bound, P value]`
- `12_time.md` — `[Row, Time value, Time unit]`

Drug and patient may be **`Row`-keyed** (if their matching stage ran) or a
**single broadcast row** (if the stage-9 / stage-10 shortcut applied):
- `09_drug_matched.md` — `[Row, Drug name, Analyte, Specimen]`, or a single
  `[Drug name, Analyte, Specimen]` to broadcast to every row.
- `10_patient_matched.md` — `[Row, Population, Pregnancy stage,
  Pediatric/Gestational age, Subject N]`, or a single cohort to broadcast.

Read these files now.

## Procedure
1. For each `## Sub-table N`, take the `Row` keys from `08_type_unit.md` as the
   authoritative row set. Confirm that `11_param_values.md` and `12_time.md`
   have the **exact same set of `Row` values** for that sub-table (and any
   `Row`-keyed drug/patient table too). If a key set disagrees — a missing,
   extra, or duplicated `Row` — **do not guess**: report the mismatch to the
   user and stop (a prior stage broke the join key).
2. For each `Row` key, gather that row's fields from each input **by matching
   the `Row` value** (not by line position). For a broadcast drug/patient table
   (single row, no `Row`), use that single combination for every row.
3. Assemble each row's columns left-to-right in this order:
   `drug → patient → type/unit → values → time`. **Drop the `Row` helper
   column** — it is a join key, not an output column.
4. Stack all sub-tables' rows into one table.
5. **Drop fully-empty rows** (rows that are entirely `N/A`/blank).
6. **Rename** two columns:
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
- column order matches the list above exactly (19 columns),
- the `Row` helper column does **not** appear in the output (it was a join key),
- row count equals the per-sub-table row counts minus any all-empty rows
  removed,
- the `Main value` / `Statistics type` headers no longer appear (they were
  renamed).

Then run the validation and verification passes described in `SKILL.md`.
