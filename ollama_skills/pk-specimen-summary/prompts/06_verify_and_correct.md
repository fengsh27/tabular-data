# Stage 06 — Verification + correction

This stage is the quality gate after cleanup. It follows the **shared** procedure
in `verify_and_correct.md` (deterministic provenance check
→ adversarial semantic check → bounded ≤2-round correction loop). Read that file
and follow it, with the parameters below filled in for the PK specimen summary
schema.

`<scratch>` below is the run directory `./.pk_specimen_summary_scratch/<pmid>/`.

## Parameters for the shared procedure
- **`<FINAL_CSV>`** = `<scratch>/05_final.csv`
- **`<SOURCE FILES>`** = `<scratch>/inputs.md` (title + full text — the source is
  **prose**, there is no markdown source table)
- **`<VALUE-COLUMNS>`** = `Sample N,Population N,Sample time`
- **`<LABEL-COLUMNS>`** = *(none — see below)*
- **`<CAUTION-COLUMNS>`** = `Sample N,Population N`
  (counts read from prose, often not verbatim isolated numbers, so a provenance
  finding on them may flag a *correct* value. Per the shared rule, treat findings
  on these as Step-B judgments, not automatic corrections.)

## Existence-only provenance (full-text source)
The source is prose, not a table, so run the script in **existence-only** mode —
**omit `--attribution`**. Note `Sample time` may be a comma-list or range
(e.g. `0, 2, 4` or `0-2`); the script splits it into component numbers and checks
each appears in the text.

```
python scripts/verify_provenance.py \
  <scratch>/05_final.csv \
  <scratch>/inputs.md \
  --value-columns "Sample N,Population N,Sample time" \
  --json
```

## Step B emphasis for this schema
Read adversarially against the full text and confirm, per row:
- the `Specimen` and its `Sample N` are what the text reports for that cohort,
- the `Sample time` + `Time unit` are the sampling times stated (not converted),
- the sampling is attributed to the right `Population` / `Population N`,
- `Pediatric/Gestational age` was not inferred from a sampling time.

## Output
Write the verification report to `<scratch>/06_verification_report.md` and
overwrite `<scratch>/05_final.csv` with the corrected table, exactly as the shared
procedure's "Output of this stage" section specifies. Distinguish a verified-clean
result from one with caveats, and list every unresolved finding.
