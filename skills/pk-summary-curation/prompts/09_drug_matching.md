# Stage 09 — Drug matching

This file is loaded by `SKILL.md` during the PK summary curation procedure.
**Only run this prompt if the shortcut in SKILL.md does not apply** (i.e. the
drug table has more than one row). With a single drug combination, assign it to
every row directly without reasoning.

## What you are doing
Attach one `[Drug name, Analyte, Specimen]` combination to each value-bearing
row of the sub-table, by matching each row to the best entry in the drug table.

## Inputs (read from the scratch directory)
- `07_subtables.md` — the per-parameter sub-tables (the rows to label).
- `01_drug_table.md` — the unique `[Drug name, Analyte, Specimen]` combinations.
- `05_param_aligned.md` — the aligned main table, for context.
- `inputs.md` — caption + title (names the drug, e.g. lorazepam).

Read these files now.

## Procedure (per sub-table)
1. Process **every row** of the sub-table, keeping each row's `Row` join-key
   value — the output must have exactly the same `Row` values as the sub-table.
2. For each row, find the **single best-matching** row in `01_drug_table.md`:
   - First locate the corresponding row in the main table by **row index**
     (sub-table row *i* ↔ main-table row *i*); that row gives more context.
   - Then choose the drug combination whose Specimen / Analyte best fits that
     row and the caption.
3. Because the sub-table preserves main-table row order, if a row is ambiguous
   you may infer its match from the row immediately before or after it.
4. If no legitimate match exists after thorough evaluation, assign `N/A` for
   that row (last resort only).

## Reasoning then answer
For each row, state which drug-table combination you matched and why, then
produce the result.

## Output of this stage
For each sub-table, a four-column markdown table — the `Row` join key (copied
verbatim from the sub-table) plus the matched combination, one row per
sub-table row:

```
| Row | Drug name | Analyte | Specimen |
| --- | --- | --- | --- |
| <row> | <matched drug> | <matched analyte> | <matched specimen> |
```

Use the same `## Sub-table N` headings as `07_subtables.md`.

Before continuing, sanity-check:
- each output table has the **same `Row` values** as its sub-table (same set,
  same order),
- every `[Drug name, Analyte, Specimen]` row appears in `01_drug_table.md`
  (or is `N/A`).

If that check fails, redo this stage once.

Then **write the result to `09_drug_matched.md`** in the scratch directory.
Stage 13 (assembly) reads that file.

## Note on the shortcut
For `16143486_table_4` the drug table has **two** rows (Cord blood, Maternal
blood specimens), so the shortcut does *not* apply and this matching runs:
each parameter row is matched to the cord-blood or maternal-blood combination
according to which specimen its parameter describes.
