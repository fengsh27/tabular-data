# Stage 04 — Assembly

This file is loaded by `procedure.md`. It joins the context tags and the parsed
values into the 12 **working-name** columns the cleanup script expects.

## Inputs (read from the scratch directory)
- `03_study_info.md` — `[Characteristic, Exposure, Outcome]` (Stage 3).
- `02_param_values.md` — `[Main value, Main value unit, Statistics type, Variation
  type, Variation value, Interval type, Lower bound, Upper bound, P value]`
  (Stage 2).

Read these now.

## How to assemble
Both tables describe the **same rows in the same order** (one row per retained
value). This is a **positional horizontal join**, row *i* of each forming output
row *i*:

1. Confirm both tables have the **same row count**. If not, do not guess — note
   the mismatch and report it; align by position as far as possible.
2. Build each output row in this exact column order (these are the **working
   names** — the cleanup script renames them in stage 5):
   `Characteristic, Exposure, Outcome` (Stage 3) → `Main value, Main value unit,
   Statistics type, Variation type, Variation value, Interval type, Lower bound,
   Upper bound, P value` (Stage 2).

That yields the 12 working columns:
`Characteristic, Exposure, Outcome, Main value, Main value unit, Statistics type,
Variation type, Variation value, Interval type, Lower bound, Upper bound, P value`.

Do **not** drop, dedupe, rename, or reorder beyond this — stage 5's script does the
business-rule cleanup, the renames (`Main value`→`Parameter value`, etc.), and the
final reorder.

## Output of this stage
Write the assembled 12-column (working-name) table to `04_assembled.csv` (CSV with
the header row exactly as above). Stage 5 cleans it.

Before continuing, sanity-check:
- exactly the 12 working-name columns in the order above,
- row count equals the stage tables',
- you did **not** rename to the final `Parameter *` names — that is stage 5's job.
