# Stage 02 — Patient information

This file is loaded by `procedure.md` during the PK individual curation procedure.

## What you are doing
Extract the unique patient/subject groupings the table reports, as rows of
`[Patient ID, Population, Pregnancy stage]`. Because this is **individual** data,
the key field is the per-subject **Patient ID**, not a subject count.

## Inputs (read from the scratch directory)
- `00_markdown_table.md` — the source PK table in markdown (Stage 0 output).
- `inputs.md` — the caption, footnotes, and paper title, verbatim.

Read these files now.

## How to extract
Examine the table row by row and column by column, and list every unique
combination of:

1. **Patient ID** — an identifier of a **unique individual subject**.
   - Use the **exact text** as it appears in the table (e.g. `1`, `3`, `Case 4`).
   - **Normally a `Patient ID` column is already present** — Stage 0c injects one
     (inferring from the full text) when the original table lacked it, so read it
     from the table.
   - If, unexpectedly, there is still no `Patient ID` column and no other
     per-subject identifier, do **not** fabricate ids — note that Stage 0c did
     not establish a Patient ID (likely `needs_full_text` in `00c_patient_id.md`)
     and stop, asking for the full text. See procedure.md's error-handling rules.
2. **Population** — the age/demographic group (adult, neonate, pregnant women,
   …). `"N/A"` if not stated or reasonably inferable.
3. **Pregnancy stage** — pregnancy-related timing (trimester, delivery,
   postpartum, …). `"N/A"` if not applicable.

Every combination must be supported by the table or caption. Prefer inference
from explicit context over `"N/A"`; use `"N/A"` only when inference is not
reasonable.

## Reasoning then answer
Explain how you identified the Patient ID column (or the unit you inferred) and
any Population / Pregnancy stage signals, then produce the table.

## Output of this stage
A markdown table of the unique combinations, columns exactly
`Patient ID`, `Population`, `Pregnancy stage`:

```
| Patient ID | Population | Pregnancy stage |
| --- | --- | --- |
| 1 | N/A | N/A |
| 3 | N/A | N/A |
```

Before continuing, sanity-check:
- columns are exactly `Patient ID, Population, Pregnancy stage`,
- one row per **unique** combination (no duplicate Patient IDs),
- Patient IDs are taken verbatim from the table (or a clearly-inferred unit).

If that check fails, redo this stage once.

Then **write the result to `02_patient_table.md`** in the scratch directory.
Stage 3 reads that file.
