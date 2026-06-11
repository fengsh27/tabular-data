# Stage 05 — Verification + correction

This stage is the quality gate after assembly. It follows the **shared** procedure
in `verify_and_correct.md` (deterministic provenance check
→ adversarial semantic check → bounded ≤2-round correction loop). Read that file
and follow it, with the parameters below filled in for the PK population summary
schema.

`<scratch>` below is the run directory `./.pk_population_summary_scratch/<pmid>/`.

## Parameters for the shared procedure
- **`<FINAL_CSV>`** = `<scratch>/04_final.csv`
- **`<SOURCE FILES>`** = `<scratch>/inputs.md` (title + full text — the source is
  **prose**, there is no markdown source table)
- **`<VALUE-COLUMNS>`** = `Characteristic value,Variation value,Lower bound,Upper bound,Subject N`
- **`<LABEL-COLUMNS>`** = *(none — see below)*
- **`<CAUTION-COLUMNS>`** = `Subject N`
  (a count read from prose, often not a verbatim isolated number, so a provenance
  finding on it may flag a *correct* value. Per the shared rule, treat `Subject N`
  findings as Step-B judgments, not automatic corrections.)

## Existence-only provenance (full-text source)
The source is prose, not a table, so run the script in **existence-only** mode —
**omit `--attribution`**. Note `Characteristic value` may be a raw ratio like
`4/5/4/3`; the script splits it into component numbers and checks each appears in
the text.

```
python scripts/verify_provenance.py \
  <scratch>/04_final.csv \
  <scratch>/inputs.md \
  --value-columns "Characteristic value,Variation value,Lower bound,Upper bound,Subject N" \
  --json
```

## Step B emphasis for this schema
Read adversarially against the full text and confirm, per row:
- the `Characteristic value` + `Statistics type` + `Variation`/`Interval` are what
  the text reports for that characteristic,
- the `Characteristic unit` matches,
- the characteristic is attributed to the right `Population` / `Subject N`,
- `Characteristic` is a genuine population characteristic, never a PK parameter,
- `Pediatric/Gestational age` was not inferred from a measurement time.

## Output
Write the verification report to `<scratch>/05_verification_report.md` and
overwrite `<scratch>/04_final.csv` with the corrected table, exactly as the shared
procedure's "Output of this stage" section specifies. Distinguish a verified-clean
result from one with caveats, and list every unresolved finding.
