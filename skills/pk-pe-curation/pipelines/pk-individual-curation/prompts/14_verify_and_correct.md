# Stage 14 — Verification + correction

This stage is the quality gate after row cleanup. It follows the **shared**
procedure in `curation-common/verify_and_correct.md` (deterministic
provenance check → adversarial semantic check → bounded ≤2-round correction
loop). Read that file and follow it, with the parameters below filled in for the
PK-individual schema.

Run it **once per curated table**, in that table's `table_<n>/` sub-directory
(every table — including the sole table of a single-table run — is nested; see
"Always nest" in procedure.md). `<scratch>` below is that `table_<n>/` directory.

## Parameters for the shared procedure
- **`<FINAL_CSV>`** = `<scratch>/13_final.csv`
- **`<SOURCE FILES>`** = `<scratch>/00_markdown_table.md <scratch>/inputs.md`
- **`<VALUE-COLUMNS>`** = `Parameter value,Time value`
- **`<LABEL-COLUMNS>`** = `Patient ID,Parameter type,Analyte,Specimen,Population`
- **`<CAUTION-COLUMNS>`** = `Patient ID`
  (often an inferred unique unit rather than a verbatim table number, so a
  provenance finding on it may flag a *correct* value. Per the shared rule, treat
  `Patient ID` findings as semantic judgments, not automatic corrections.)

So the concrete script invocation is:

```
python curation-common/scripts/verify_provenance.py \
  <scratch>/13_final.csv \
  <scratch>/00_markdown_table.md \
  <scratch>/inputs.md \
  --value-columns "Parameter value,Time value" \
  --attribution \
  --label-columns "Patient ID,Parameter type,Analyte,Specimen,Population" \
  --json
```

Note on what attribution does and does not catch here: it flags a value that
does not appear under the **parameter** the row claims (it best-matches a
different parameter column). It does **not** reliably catch a value swapped
between **two subjects of the same parameter** — both share that parameter's
source column. Confirming each value belongs to the right **Patient ID** is the
job of the Step-B semantic check; do it carefully, subject by subject.

## Output
Write the verification report to `<scratch>/14_verification_report.md` and
overwrite `<scratch>/13_final.csv` with the corrected table, exactly as the
shared procedure's "Output of this stage" section specifies. The semantic check
(Step B) should confirm, in particular, that each value belongs to the Patient
ID and parameter the row claims, and that the unit matches.
