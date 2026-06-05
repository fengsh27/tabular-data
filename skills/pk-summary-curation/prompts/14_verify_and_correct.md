# Stage 14 — Verification + correction

This stage is the quality gate after assembly. It follows the **shared**
procedure in `skills/curation-common/verify_and_correct.md` (deterministic
provenance check → adversarial semantic check → bounded ≤2-round correction
loop). Read that file and follow it, with the parameters below filled in for the
PK-summary schema.

Run it **once per curated table**, in that table's `table_<n>/` sub-directory
(every table — including the sole table of a single-table run — is nested; see
"Always nest" in SKILL.md). `<scratch>` below is that `table_<n>/` directory.

## Parameters for the shared procedure
- **`<FINAL_CSV>`** = `<scratch>/13_final.csv`
- **`<SOURCE FILES>`** = `<scratch>/00_markdown_table.md <scratch>/inputs.md`
- **`<VALUE-COLUMNS>`** =
  `Parameter value,Variation value,Lower bound,Upper bound,Subject N,Time value,P value`
- **`<LABEL-COLUMNS>`** =
  `Parameter type,Analyte,Specimen,Population,Pregnancy stage`
- **`<CAUTION-COLUMNS>`** = `Subject N`
  (often taken from the caption — e.g. "n = 8" — or legitimately refined per
  parameter, so a provenance finding on it may be a *correct* value. Per the
  shared rule, treat `Subject N` findings as semantic judgments, not automatic
  corrections.)

So the concrete script invocation is:

```
python skills/curation-common/scripts/verify_provenance.py \
  <scratch>/13_final.csv \
  <scratch>/00_markdown_table.md \
  <scratch>/inputs.md \
  --value-columns "Parameter value,Variation value,Lower bound,Upper bound,Subject N,Time value,P value" \
  --attribution \
  --label-columns "Parameter type,Analyte,Specimen,Population,Pregnancy stage" \
  --json
```

## Output
Write the verification report to `<scratch>/14_verification_report.md` and
overwrite `<scratch>/13_final.csv` with the corrected table, exactly as the
shared procedure's "Output of this stage" section specifies. The semantic check
(Step B) should confirm, in particular, that each row's `Parameter statistic`
(Mean / Median / …) and `Parameter unit` match what the source reports.
