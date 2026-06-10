# Stage 04 — Verification + correction

This stage is the quality gate after assembly. It follows the **shared** procedure
in `curation-common/verify_and_correct.md` (deterministic provenance check
→ adversarial semantic check → bounded ≤2-round correction loop). Read that file
and follow it, with the parameters below filled in for the PE study info schema.

`<scratch>` below is the run directory `./.pe_study_info_scratch/<pmid>/`.

## Parameters for the shared procedure
- **`<FINAL_CSV>`** = `<scratch>/03_final.csv`
- **`<SOURCE FILES>`** = `<scratch>/inputs.md` (title + full text — the source is
  **prose**, there is no markdown source table)
- **`<VALUE-COLUMNS>`** = `Subject N`
- **`<LABEL-COLUMNS>`** = *(none — see below)*
- **`<CAUTION-COLUMNS>`** = `Subject N`
  (a count read from prose, often not a verbatim isolated number, so a provenance
  finding on it may flag a *correct* value. Per the shared rule, treat `Subject N`
  findings as Step-B judgments, not automatic corrections.)

## Existence-only provenance (full-text source)
The source is prose, not a table, so run the script in **existence-only** mode —
**omit `--attribution`**. Almost every column here is free text; the only numeric
value to check is `Subject N`.

```
python curation-common/scripts/verify_provenance.py \
  <scratch>/03_final.csv \
  <scratch>/inputs.md \
  --value-columns "Subject N" \
  --json
```

## Step B emphasis for this schema (mostly semantic)
This schema is mostly free text, so Step B carries the weight. Read adversarially
against the full text and confirm:
- `Study type` and `Study design` match what the article describes,
- `Inclusion criteria`, `Exclusion criteria`, and `Outcomes` are the article's
  **exact wording** (not paraphrased or invented),
- `Population` / `Pregnancy stage` are correctly categorized,
- `Drug name`, `Data source`, and `Subject N` are what the article reports.

## Output
Write the verification report to `<scratch>/04_verification_report.md` and
overwrite `<scratch>/03_final.csv` with the corrected row, exactly as the shared
procedure's "Output of this stage" section specifies. Distinguish a verified-clean
result from one with caveats, and list every unresolved finding.
