# Verify + correct (generic, shared)

This is the shared quality gate for the curation skills. A pipeline's final
verification stage points here and supplies four things:
- **`<FINAL_CSV>`** — the assembled curated CSV for one table (e.g.
  `<scratch>/13_final.csv`).
- **`<SOURCE FILES>`** — that table's source: the markdown table + the caption
  /inputs file (e.g. `<scratch>/00_markdown_table.md <scratch>/inputs.md`).
- **`<VALUE-COLUMNS>`** — the **numeric** columns whose values must trace to the
  source (the calling stage lists them).
- **`<LABEL-COLUMNS>`** — the discriminator columns used for attribution (the
  calling stage lists them).
- **`<CAUTION-COLUMNS>`** — columns whose findings are *flags for judgment, not
  auto-corrections* (caption-sourced / inferred fields; e.g. `Subject N`,
  `Patient ID`). May be empty.

`<scratch>` is the table's own stage-output directory. This stage runs **once
per curated table**, each against its own source.

## What you are doing
Confirm `<FINAL_CSV>` is actually supported by the source, then correct what is
wrong — in a **bounded** loop, then stop and report what remains. Verification
has a **deterministic** half (the script) and a **semantic** half (your
judgment). Use both.

## Step A — Deterministic provenance check (run the script)
Run with `--attribution` so it checks both existence and attribution:

```
python skills/curation-common/scripts/verify_provenance.py \
  <FINAL_CSV> <SOURCE FILES> \
  --value-columns "<VALUE-COLUMNS>" \
  --attribution --label-columns "<LABEL-COLUMNS>" \
  --json
```

Scope to **numeric** columns with `--value-columns` so normalized categorical
labels that embed a number (e.g. a `Pregnancy stage` of "Trimester 1") are not
flagged. The script reports **two** kinds of finding; know what each proves:
- **`findings` (existence)** — the number appears nowhere in the source: a real
  failure (hallucinated, mistyped, mis-transcribed). Existence catches
  *fabrication*, not misplacement — a value copied to the wrong row still
  "exists", so existence alone cannot confirm a row.
- **`attribution.findings` (misattribution)** — the number exists but **not
  under the label this row claims** (it best matches a different column/row).
  Catches a value swapped between cohorts/specimens. Attribution is best-effort
  (only fires for rows with distinctive labels and table-sourced numbers), so a
  clean pass is reassuring but not a proof — Step B still matters.

**`<CAUTION-COLUMNS>` are never auto-corrected from a script finding.** Those
fields (e.g. a subject count from the caption, or an inferred patient id) are
routinely *not* verbatim table numbers, so a finding on them may flag a correct
value. Treat every finding on a caution column as a **Step-B judgment only**:
change it only if your semantic reading of the source shows it is genuinely
wrong; otherwise leave it and report it as unresolved. Do not "correct" it just
to silence the script.

## Step B — Semantic attribution check (your judgment)
The script's existence check proves a number *exists*; its attribution mode is a
coarse token-overlap signal, not a proof; neither settles borderline attribution
or non-numeric correctness. For each curated row, verify against the source:
- the **value** and (if the schema has one) its **statistic** are what the
  source reports for that parameter,
- the row's **cohort / subject** is the one the source associates with the value,
- the **Analyte** and **Specimen** attribution matches,
- the **unit** matches.

Work **adversarially**: actively try to find the row the source does *not*
justify. If unsure a row is supported, flag it rather than pass it.

## Step C — Bounded correction loop
1. If there are **no** findings, the table passes — go to Output.
2. Otherwise produce a corrected `<FINAL_CSV>` that fixes **only the flagged
   rows**, taking every corrected value directly from the source (no
   calculation, no invention). Do not touch rows that passed. Respect the
   `<CAUTION-COLUMNS>` rule above.
3. Re-run Step A and re-do Step B on the corrected table.
4. Repeat at most **2 correction rounds total**. If findings remain after the
   second round, **stop**.

Never silently drop a row to make a check pass. If a row genuinely has no
support in the source, leave it and report it as unresolved.

## Output of this stage
- The corrected `<FINAL_CSV>` (overwrite in place).
- A verification report file in `<scratch>` containing, per round: how many
  numbers were checked / unsupported / misattributed (from the script), the
  semantic findings and what you changed, and any **unresolved** findings left
  after the bound.
- In your reply, present the final table and **call out every unresolved finding
  explicitly**. A verified-clean result and a result-with-caveats must be
  clearly distinguished.

Before finishing, sanity-check:
- the script exits clean (0 unsupported, 0 misattributed) **or** every remaining
  finding is listed as unresolved in the report,
- no row was dropped solely to pass verification,
- the loop ran at most 2 rounds.

## Note for weaker (open) models
The semantic check is partly circular when the model that produced an error also
judges it, so lean on the deterministic script for anything numeric and keep the
semantic pass strictly adversarial. Report uncertainty honestly rather than
rubber-stamping.
