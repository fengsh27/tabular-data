# Stage 06 — Verification + correction

This stage is the quality gate after cleanup. It follows the **shared** procedure
in `curation-common/verify_and_correct.md` (deterministic provenance check
→ adversarial semantic check → bounded ≤2-round correction loop). Read that file
and follow it, with the parameters below filled in for the PK population
individual schema.

`<scratch>` below is the run directory `./.pk_population_individual_scratch/<pmid>/`.

## Parameters for the shared procedure
- **`<FINAL_CSV>`** = `<scratch>/05_final.csv`
- **`<SOURCE FILES>`** = `<scratch>/inputs.md` (title + full text — the source is
  **prose**, there is no markdown source table)
- **`<VALUE-COLUMNS>`** = `Characteristic value`
- **`<LABEL-COLUMNS>`** = *(none — see below)*
- **`<CAUTION-COLUMNS>`** = `Patient ID`
  (often inferred from the case structure rather than a verbatim isolated number,
  so a provenance finding on it may flag a *correct* value. Per the shared rule,
  treat `Patient ID` findings as Step-B judgments, not automatic corrections.)

## Existence-only provenance (full-text source)
The source is prose, not a table, so run the script in **existence-only** mode —
**omit `--attribution`**. `Characteristic value` may be a raw ratio like
`4/5/4/3`; the script splits it into component numbers and checks each appears in
the text.

```
python curation-common/scripts/verify_provenance.py \
  <scratch>/05_final.csv \
  <scratch>/inputs.md \
  --value-columns "Characteristic value" \
  --json
```

## Step B emphasis for this schema
Read adversarially against the full text and confirm, per row:
- the `Characteristic value` + `Characteristic unit` are what the text reports
  **for that `Patient ID`**,
- the characteristic is attributed to the right patient (no value swapped between
  two patients — the existence check cannot catch that),
- `Characteristic` is a genuine characteristic, never a PK parameter,
- `Pediatric/Gestational age` was not inferred from a measurement time.

## Output
Write the verification report to `<scratch>/06_verification_report.md` and
overwrite `<scratch>/05_final.csv` with the corrected table, exactly as the shared
procedure's "Output of this stage" section specifies. Distinguish a verified-clean
result from one with caveats, and list every unresolved finding.
