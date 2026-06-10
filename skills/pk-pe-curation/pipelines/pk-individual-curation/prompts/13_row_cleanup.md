# Stage 13 — Row cleanup (deterministic script)

This file is loaded by `procedure.md` after assembly. It applies the individual
pipeline's mechanical normalization rules with a **bundled deterministic
script** — do not apply these rules by hand, the script is the source of truth.

## Inputs (read from the scratch directory)
- `12_assembled.csv` — the assembled 12-column table from stage 12.

## Run the cleanup script

```
python pipelines/pk-individual-curation/scripts/clean_individual_rows.py \
  <scratch>/12_assembled.csv -o <scratch>/13_final.csv
```

`<scratch>` is this table's `table_<n>/` sub-directory. The script:
1. drops any row containing `ERROR`;
2. blanks the `Time value` when the `Time unit` is a long duration (weeks /
   months / years) — those are ages, not sampling times;
3. couples time: if `Time value` or `Time unit` is `N/A`, sets both `N/A`;
4. if `Parameter value` is `N/A`, sets `Parameter type` and `Parameter unit`
   `N/A`;
5. blanks the time for `Cmax` / `Tmax` / `Cavg` (no sampling time);
6. normalizes blanks/`unknown`/`nan` → `N/A`;
7. **drops rows whose `Parameter value` is `N/A`** (nothing measured);
8. drops duplicate rows;
9. orders columns with `Patient ID` first.

It needs only the Python standard library (no extra install).

## Output of this stage
The script writes `13_final.csv` (the 12-column individual result). Read it back
and present it to the user as a markdown table.

Before continuing, sanity-check the written file:
- 12 columns, `Patient ID` first, in the schema order;
- no row has `Parameter value` == `N/A`;
- no duplicate rows.

Then proceed to stage 14 (verification + correction).
