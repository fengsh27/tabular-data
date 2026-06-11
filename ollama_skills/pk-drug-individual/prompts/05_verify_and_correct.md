# Stage 05 — Verification + correction

This stage is the quality gate after assembly. It follows the **shared**
procedure in `verify_and_correct.md` (deterministic
provenance check → adversarial semantic check → bounded ≤2-round correction
loop). Read that file and follow it, with the parameters below filled in for the
PK drug individual schema.

`<scratch>` below is the run directory `./.pk_drug_individual_scratch/<pmid>/`.

## Parameters for the shared procedure
- **`<FINAL_CSV>`** = `<scratch>/04_final.csv`
- **`<SOURCE FILES>`** = `<scratch>/inputs.md` (the title + full text — the source
  is **prose**, there is no markdown source table)
- **`<VALUE-COLUMNS>`** = `Dose amount`
- **`<LABEL-COLUMNS>`** = *(none — see below)*
- **`<CAUTION-COLUMNS>`** = `Patient ID`
  (often inferred from the case structure rather than a verbatim isolated number,
  so a provenance finding on it may flag a *correct* value. Per the shared rule,
  treat `Patient ID` findings as Step-B judgments, not automatic corrections.)

## Existence-only provenance (full-text source)
The source is the paper's prose, not a table, so run the script in
**existence-only** mode — **omit `--attribution`** (there is no source table for
the attribution check to match against). The concrete invocation:

```
python scripts/verify_provenance.py \
  <scratch>/04_final.csv \
  <scratch>/inputs.md \
  --value-columns "Dose amount" \
  --json
```

This confirms every `Dose amount` number actually appears somewhere in the full
text (catches hallucinated or mistyped doses). It cannot confirm a dose belongs
to the *right* patient — that is the job of the Step-B semantic check.

## Step B emphasis for this schema
Read adversarially against the full text and confirm, per row:
- the `Dose amount` + `Dose unit`, `Dose frequency`, `Dose schedule`, and
  `Dose route` are what the text reports **for that `Patient ID`**,
- the dosing is attributed to the right patient (no value swapped between two
  patients/cases — the existence check cannot catch that),
- `Pediatric/Gestational age` was not inferred from a measurement/dosing time.

## Output
Write the verification report to `<scratch>/05_verification_report.md` and
overwrite `<scratch>/04_final.csv` with the corrected table, exactly as the
shared procedure's "Output of this stage" section specifies. Distinguish a
verified-clean result from one with caveats, and list every unresolved finding.
