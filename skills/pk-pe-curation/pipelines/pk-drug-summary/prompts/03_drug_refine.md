# Stage 03 — Drug / dose refine

This file is loaded by `procedure.md`. It refines the stage-1 dosing rows into the
detailed drug columns of the output schema.

## What you are doing
For **each row** of the stage-1 table, decompose the dosing into
`[Drug/Metabolite name, Dose amount, Dose unit, Dose frequency, Dose schedule,
Dose route]` — one output row per stage-1 row, in the same order.

## Inputs (read from the scratch directory)
- `01_drug_info.md` — the stage-1 dosing table (the rows to refine).
- `inputs.md` — title + full text, to resolve route/schedule the table doesn't
  spell out.

Read these now.

## How to refine each row
- **Drug/Metabolite name** — the drug or metabolite (carry from stage 1).
- **Dose amount** — the numeric amount only: a value, comma list, or range
  (e.g. `5`, `1,2,3,4`, `0.01 - 0.05`). **Strip the unit off** — if stage 1 had
  `5 mg`, the amount is `5`.
- **Dose unit** — the unit that was attached to the amount (e.g. `mg`, `mg/kg`).
- **Dose frequency** — times taken (e.g. `Single`, `Multiple`, `3`).
- **Dose schedule** — the timing/interval (e.g. `once a day`, `twice a day`,
  `every 8 hours`).
- **Dose route** — route of administration: `Oral`, `Intravenous (IV)`,
  `Intramuscular (IM)`, `Subcutaneous (SC)`, `Epidural`, `Infusion`, etc.

Use `"N/A"` for any field that cannot be reasonably inferred from the text.

## Hard rules
- Exactly one output row per stage-1 row, in the **same order** — no more, no
  less, no reordering.
- Take values from the text; do not calculate or invent.

## Reasoning then answer
Explain how you split amount vs. unit and where route/schedule came from, then
produce the table.

## Output of this stage
A markdown table with columns exactly
`Drug/Metabolite name, Dose amount, Dose unit, Dose frequency, Dose schedule, Dose route`,
one row per stage-1 row.

Before continuing, sanity-check:
- row count equals `01_drug_info.md`'s row count, same order,
- `Dose amount` carries no unit and `Dose unit` carries no number.

If a check fails, redo this stage once. Then **write the result to
`03_drug_refined.md`** in the scratch directory. Stage 4 reads it.
