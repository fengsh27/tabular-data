# Stage 05 — Row cleanup (deterministic script)

This file is loaded by `procedure.md`. It runs the bundled, deterministic cleanup
script on the assembled table — **do not** do this cleanup by hand.

## What the script does
`curation-common/scripts/clean_population_individual_rows.py` drops any row
whose `Characteristic value` is blank or `N/A` (after trimming, normalizing spaced
slashes like `4 / 5` → `4/5`, and upper-casing — so `N / A` is also dropped).
Rows that survive keep their original value verbatim; original order is preserved.

## Run it
```
python curation-common/scripts/clean_population_individual_rows.py \
  <scratch>/04_assembled.csv -o <scratch>/05_final.csv
```
where `<scratch>` is `./.pk_population_individual_scratch/<pmid>/`.

If the script cannot run (e.g. Python unavailable), drop the blank/`N/A`
`Characteristic value` rows by hand and note that you did so — but prefer the
script.

## Output of this stage
`05_final.csv` (the script writes it). This is the 9-column result that stage 6
verifies and corrects in place.

Before continuing, sanity-check:
- `05_final.csv` has the same 9 columns as `04_assembled.csv`,
- no surviving row has a blank/`N/A` `Characteristic value`,
- row count is ≤ the input (cleanup only drops, never adds).
