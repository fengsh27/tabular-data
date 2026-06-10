# Stage 04 — Summary-data deletion

This file is loaded by `procedure.md` during the PK individual curation procedure.

## What you are doing
Keep only the **individual, per-subject** rows; delete the **summary / aggregate**
rows. This is the inverse of the pk-summary pipeline's stage 4 (which keeps the
summary rows and drops the individuals).

## Inputs (read from the scratch directory)
- `00_markdown_table.md` — the source PK table in markdown.

Read this file now.

## What to delete vs keep
- **Delete** rows that report summary statistics / aggregated or group-level
  values rather than one subject — e.g. `N = …`, `Mean ± SD`, `Median`,
  `Range`, `Geometric mean`, a "Total"/"Overall" row, or a cohort-level summary.
  A row with no association to a specific individual is a summary row.
- **Keep** every row that belongs to a specific individual — i.e. a row carrying
  (or grouped under) a Patient ID. Keep it **even if some of its cells are
  non-numeric or missing**; it is still part of that subject's record.

When a row's status is ambiguous, look at **adjacent rows** to infer the
grouping. A block of rows under a single subject id (e.g. an id row followed by
that subject's per-drug rows) is all individual-level — keep the whole block.

## Reasoning then answer
List which rows you are dropping and why (name the summary marker that
identifies each), confirm the rest are per-subject, then produce the table.

## Output of this stage
The same table with the summary rows removed and all individual rows kept,
columns unchanged. If the table is already all-individual, return it unchanged
and say so.

Before continuing, sanity-check:
- no aggregate/summary row remains (no `N=`, Mean, Median, Range, Total/Overall),
- every per-subject row is retained (none dropped for being non-numeric),
- columns are unchanged from the input.

If that check fails, redo this stage once.

Then **write the result to `04_individual_only.md`** in the scratch directory.
Stage 5 reads that file.
