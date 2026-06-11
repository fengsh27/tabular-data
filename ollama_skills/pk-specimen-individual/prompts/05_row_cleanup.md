# Stage 05 — Row cleanup (deterministic script)

This file is loaded by `procedure.md`. It runs the bundled, deterministic cleanup
script on the assembled table — **do not** do this cleanup by hand; the rules are
mechanical and must match the legacy pipeline exactly.

## What the script does
`scripts/clean_specimen_rows.py` applies two rules, in
order (both no-op unless every `Sample N` is an integer):
1. **Remove summed-total row** — drops the row whose `Sample N` equals exactly
   half the column total (the redundant total when individual parts are present).
2. **Keep max Sample N** — collapses rows identical across all columns *except*
   `Sample N`, `Population N`, and `Note`, keeping the one with the largest
   `Sample N` (ties → earliest). Original row order is preserved. (The individual
   schema has no `Population N`, so in practice the comparison excludes only
   `Sample N` and `Note` — the same script serves both specimen skills.)

## Run it
```
python scripts/clean_specimen_rows.py \
  <scratch>/04_assembled.csv -o <scratch>/05_final.csv
```
where `<scratch>` is `./.pk_specimen_individual_scratch/<pmid>/`.

If the script cannot run (e.g. Python unavailable), apply the two rules above by
hand and note that you did so — but always prefer the script.

## Output of this stage
`05_final.csv` (the script writes it). This is the 9-column result that stage 6
verifies and corrects in place.

Before continuing, sanity-check:
- `05_final.csv` has the same 9 columns as `04_assembled.csv`,
- row count is ≤ the input (cleanup only drops/merges, never adds).
