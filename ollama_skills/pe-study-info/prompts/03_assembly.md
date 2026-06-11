# Stage 03 — Assembly

This file is loaded by `procedure.md`. It joins the two one-row stage tables and
reorders into the final 10-column schema.

## Inputs (read from the scratch directory)
- `02_design_refined.md` — `[Population, Inclusion criteria, Exclusion criteria,
  Pregnancy stage, Subject N, Drug name, Outcomes]` (Stage 2).
- `01_design_info.md` — `[Study type, Study design, Data source]` (Stage 1).

Read these now.

## How to assemble
Both tables are a **single row** describing the same study. Combine their columns
and reorder into this exact 10-column order:

`Study type` (S1), `Population` (S2), `Study design` (S1), `Pregnancy stage` (S2),
`Drug name` (S2), `Data source` (S1), `Inclusion criteria` (S2), `Exclusion
criteria` (S2), `Outcomes` (S2), `Subject N` (S2).

That yields the 10 columns:
`Study type, Population, Study design, Pregnancy stage, Drug name, Data source,
Inclusion criteria, Exclusion criteria, Outcomes, Subject N`.

## Output of this stage
Write the assembled single-row 10-column table to `03_final.csv` (CSV with the
header row exactly as above). Stage 4 verifies and corrects it in place.

Before continuing, sanity-check:
- exactly 10 columns in the schema order,
- exactly **one** data row.
