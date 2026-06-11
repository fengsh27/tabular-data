# Stage 05 — Verification + correction

This stage is the quality gate after assembly. It follows the **shared**
procedure in `verify_and_correct.md` (deterministic
provenance check → adversarial semantic check → bounded ≤2-round correction
loop). Read that file and follow it, with the parameters below filled in for the
PK drug summary schema.

`<scratch>` below is the run directory `./.pk_drug_summary_scratch/<pmid>/`.

## Parameters for the shared procedure
- **`<FINAL_CSV>`** = `<scratch>/04_final.csv`
- **`<SOURCE FILES>`** = `<scratch>/inputs.md` (the title + full text — the source
  is **prose**, there is no markdown source table)
- **`<VALUE-COLUMNS>`** = `Dose amount,Population N`
- **`<LABEL-COLUMNS>`** = *(none — see below)*
- **`<CAUTION-COLUMNS>`** = `Population N`
  (a count read from prose, often not a verbatim isolated number, so a provenance
  finding on it may flag a *correct* value. Per the shared rule, treat
  `Population N` findings as Step-B judgments, not automatic corrections.)

## Existence-only provenance (full-text source)
The source is the paper's prose, not a table, so run the script in
**existence-only** mode — **omit `--attribution`** (there is no source table for
the argmax-Jaccard attribution check to match against; running it would only add
noise). The concrete invocation:

```
python scripts/verify_provenance.py \
  <scratch>/04_final.csv \
  <scratch>/inputs.md \
  --value-columns "Dose amount,Population N" \
  --json
```

This confirms every `Dose amount` / `Population N` number actually appears
somewhere in the full text (catches hallucinated or mistyped doses/counts). It
cannot confirm a number sits with the *right* drug or population — that is the
job of the Step-B semantic check.

## Step B emphasis for this schema
Read adversarially against the full text and confirm, per row:
- the `Dose amount` + `Dose unit` are what the text reports for that
  `Drug/Metabolite name`,
- the `Dose frequency`, `Dose schedule`, and `Dose route` match the text,
- the dosing is attributed to the right `Population` (and `Population N`),
- `Pediatric/Gestational age` was not inferred from a measurement/dosing time.

## Output
Write the verification report to `<scratch>/05_verification_report.md` and
overwrite `<scratch>/04_final.csv` with the corrected table, exactly as the
shared procedure's "Output of this stage" section specifies. Distinguish a
verified-clean result from one with caveats, and list every unresolved finding.
