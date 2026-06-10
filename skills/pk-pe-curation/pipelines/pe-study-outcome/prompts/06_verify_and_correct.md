# Stage 06 — Verification + correction

This stage is the quality gate after cleanup. It follows the **shared** procedure
in `curation-common/verify_and_correct.md` (deterministic provenance check
→ adversarial semantic check → bounded ≤2-round correction loop). Read that file
and follow it, with the parameters below filled in for the PE study outcome
schema.

`<scratch>` below is the run directory `./.pe_study_outcome_scratch/<pmid>/`.

## Parameters for the shared procedure
- **`<FINAL_CSV>`** = `<scratch>/05_final.csv`
- **`<SOURCE FILES>`** = `<scratch>/00_markdown_table.md <scratch>/inputs.md`
  (the source **is a table** here — unlike the full-text PE skill)
- **`<VALUE-COLUMNS>`** = `Parameter value,Variation value,Lower bound,Upper bound,P value`
- **`<LABEL-COLUMNS>`** = `Characteristic,Exposure,Outcome`
- **`<CAUTION-COLUMNS>`** = *(none)*

## Provenance with attribution (table source)
Because the source is a **markdown table**, run the script **with
`--attribution`** — it checks both that each number exists in the table and that
it sits under the column/row the curated row's `Characteristic`/`Exposure`/
`Outcome` best matches (catching a value tagged to the wrong exposure/outcome):

```
python curation-common/scripts/verify_provenance.py \
  <scratch>/05_final.csv \
  <scratch>/00_markdown_table.md \
  <scratch>/inputs.md \
  --value-columns "Parameter value,Variation value,Lower bound,Upper bound,P value" \
  --attribution \
  --label-columns "Characteristic,Exposure,Outcome" \
  --json
```

## Step B emphasis for this schema
Read adversarially against the source table and confirm, per row:
- the `Parameter value` + `Parameter statistic` + `Parameter unit` are what the
  cell reports (unit is a unit, not `SD`),
- the value is tagged to the correct `Characteristic` / `Exposure` / `Outcome`
  (the cell's row and column headers),
- intervals sit in `Lower`/`Upper bound`, not `Variation value`,
- the `P value` is the one the table associates with this value.

## Output
Write the verification report to `<scratch>/06_verification_report.md` and
overwrite `<scratch>/05_final.csv` with the corrected table, exactly as the shared
procedure's "Output of this stage" section specifies. Distinguish a verified-clean
result from one with caveats, and list every unresolved finding.
