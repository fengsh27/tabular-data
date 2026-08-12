# Stage 0c — Ensure a Patient ID (infer from full text if missing)

This file is loaded by `procedure.md` as a **per-table preprocessing step**, run for
each selected table right after its `00_markdown_table.md` is written and before
stage 1. It guarantees the table has a per-row **Patient ID** — because much
individual PK data (especially single-patient case reports) keeps the subject
identity in the **prose**, not the table. Without this step such a table has no
Patient ID column and would be discarded at stage 2.

## Inputs (read from the scratch directory)
- `00_markdown_table.md` — this table's source in markdown (Stage 0a output).
- `inputs.md` — caption + footnotes + paper title.
- `full_text.md` — the paper's full text, at the **run root**
  (`../full_text.md` relative to a `table_<n>/` directory). Step 0 of the
  procedure copies it there from `./.paper_assets/<pmid>/paper_text.md` (the
  prepared file's name differs) or from the user's pasted text. May be absent if
  neither was available.

Read these now.

## Step 1 — Does the table already identify each subject?
Examine the table column by column. If some column already uniquely identifies
the individual subject of each row (an explicit patient / subject / case id,
volunteer number, mother–infant pair, etc.), then **no injection is needed** —
leave `00_markdown_table.md` unchanged and skip to Output, recording
`already_present`.

## Step 2 — Infer a Patient ID per row (only if Step 1 found none)
Using the **full text and caption**, infer the Patient ID for **each row**, in
table-row order:
- If the study describes a **single patient** (a case report, "we present a
  …"), assign `1` to every row.
- If it describes **multiple patients/cases**, assign consistent ids from the
  text (Case 1 → `1`, Patient 2 → `2`, …).
- If a row's subject genuinely cannot be determined, use `"N/A"` for that row.

The list of ids **must have exactly as many entries as the table has rows**.

If `full_text.md` is **absent**: only infer when the caption/title make the
subject structure clear (e.g. an explicit single-patient case report → all `1`).
Otherwise do **not** guess — leave the table unchanged, record
`needs_full_text`, and let stage 2 decide (it may stop with "no Patient ID").

## Step 3 — Inject the column (only if Step 2 produced ids)
Insert a new **first column named `Patient ID`** into the table, filled with the
inferred ids (row for row, same order), and **overwrite `00_markdown_table.md`**
with the augmented table. Every later stage then sees a real Patient ID column.

## Reasoning then answer
State whether the table already had a Patient ID; if you inferred ids, explain
the subject structure you read from the full text/caption (how many patients,
which rows map to which), then produce the result.

## Output of this stage
- If ids were injected: the rewritten `00_markdown_table.md` (with `Patient ID`
  as the first column).
- A short note `00c_patient_id.md` recording one of: `already_present`,
  `inferred` (with the id list and the single-vs-multiple-patient reasoning), or
  `needs_full_text` (no id, none injected).

Before continuing, sanity-check:
- if injected, the number of `Patient ID` values equals the table's row count,
  in the original row order;
- you did **not** drop, reorder, or alter any existing rows/columns — only
  prepended `Patient ID`.

Then proceed to stage 1. Stages 2 and 4 read the (possibly augmented)
`00_markdown_table.md`.
