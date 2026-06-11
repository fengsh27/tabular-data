# Stage 05 — Row cleanup (deterministic script)

This file is loaded by `procedure.md`. It runs the bundled, deterministic cleanup
script on the assembled table — **do not** do this cleanup by hand; the rules are
mechanical and must match the legacy `pe_study_outcome_ver2` pipeline exactly.

## What the script does
`scripts/clean_pe_outcome_rows.py` applies, in order:
- interval/statistic business rules (e.g. if the Main value equals a bound, blank
  it; if both bounds are present, set `Interval type` = `Range`; if the value is
  N/A, blank the statistic; if variation value is N/A, blank variation type);
- sentinel normalization (blank → `N/A`; `n/a`/`unknown`/`nan` → `N/A`;
  `Standard Deviation (SD)`/`s.d.` → `SD`);
- **drops rows** where none of `Main value`, `Variation type`, `Lower bound`,
  `Upper bound` contains a digit;
- renames `Main value`→`Parameter value`, `Statistics type`→`Parameter statistic`,
  `Main value unit`→`Parameter unit`, and reorders to the final 12-column schema.

## Run it
```
python scripts/clean_pe_outcome_rows.py \
  <scratch>/04_assembled.csv -o <scratch>/05_final.csv
```
where `<scratch>` is `./.pe_study_outcome_scratch/<pmid>/`.

If the script cannot run (e.g. Python unavailable), apply the rules above by hand
and note that you did so — but always prefer the script.

## Output of this stage
`05_final.csv` (the script writes it). This is the final 12-column schema that
stage 6 verifies and corrects in place.

Before continuing, sanity-check:
- `05_final.csv` has the final 12 columns (`Parameter value`, `Parameter
  statistic`, `Parameter unit`, …), not the working names,
- every surviving row has a digit in at least one of the value/variation/bound
  columns,
- row count is ≤ the input (cleanup only drops/normalizes, never adds).
