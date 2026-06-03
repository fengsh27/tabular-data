# Stage 14 — Verification + correction

This file is loaded by `SKILL.md` as the quality gate after assembly. It checks
the final curated table against the source and fixes what is wrong, in a
**bounded** loop. It supersedes the lightweight "Verification (recommended)"
note in SKILL.md.

## What you are doing
Confirm that `13_final.csv` is actually supported by the source table + caption,
then correct any unsupported rows — at most a couple of rounds, then stop and
report what remains. Verification has two halves: a **deterministic** numeric
check (a script) and a **semantic** attribution check (your judgment). Use both.

## Inputs (read from the scratch directory)
- `13_final.csv` — the assembled curated table from stage 13.
- `00_markdown_table.md` — the source PK table in markdown.
- `inputs.md` — caption + footnotes + title.

## Step A — Deterministic provenance check (run the script)
Every numeric value in the curated table must appear verbatim in the source
("no calculations" rule). Run:

```
python skills/pk-summary-curation/scripts/verify_provenance.py \
  .pk_curation_scratch/<case>/13_final.csv \
  .pk_curation_scratch/<case>/00_markdown_table.md \
  .pk_curation_scratch/<case>/inputs.md \
  --value-columns "Parameter value,Variation value,Lower bound,Upper bound,Subject N,Time value,P value" \
  --json
```

Scope the check to the **numeric** columns with `--value-columns` (above) so
that normalized categorical labels that embed numbers — e.g. a `Pregnancy
stage` of "Trimester 1" — are not falsely flagged. Each finding names a row,
column, and an unsupported number: those values were hallucinated, mistyped, or
mis-transcribed.

## Step B — Semantic attribution check (your judgment)
The script proves a number *exists* in the source; it cannot prove the number
is in the *right row*. For each curated row, verify against the source that:
- the **Parameter value** and **Parameter statistic** are what the source
  reports for that parameter (e.g. it really is a Mean, not a Median),
- the row's **cohort** (Population / Pregnancy stage / Subject N) is the one the
  source associates with that value,
- the **Analyte** and **Specimen** attribution matches the source,
- the **Parameter unit** matches.

Work **adversarially**: actively try to find the row the source does *not*
justify. If you are uncertain whether a row is supported, flag it rather than
pass it.

## Step C — Bounded correction loop
Collect the findings from A and B. Then:

1. If there are **no** findings, the table passes — go to Output.
2. Otherwise, produce a corrected `13_final.csv` that fixes **only the flagged
   rows**, taking every corrected value directly from the source (no
   calculation, no invention). Do not touch rows that passed.
3. Re-run Step A and re-do Step B on the corrected table.
4. Repeat at most **2 correction rounds total**. If findings remain after the
   second round, **stop** — do not loop further.

Never silently drop a row to make the check pass. If a row genuinely has no
support in the source, leave it and report it as unresolved.

## Output of this stage
- The corrected `13_final.csv` (overwrite in place).
- A `14_verification_report.md` in the scratch directory containing:
  - how many numbers were checked and how many were unsupported (from the
    script), per round,
  - the semantic findings and what you changed,
  - any **unresolved** findings still present after the bound was reached.
- In your reply to the user, present the final table and **call out every
  unresolved finding explicitly** — do not bury or omit them. A verified-clean
  result and a result-with-caveats must be clearly distinguished.

Before finishing, sanity-check:
- the deterministic script exits clean (0 unsupported) **or** every remaining
  unsupported number is listed as unresolved in the report,
- no row was dropped solely to pass verification,
- the correction loop ran at most 2 rounds.

## Note for weaker (open) models
The semantic check is partly circular when the same model that produced an error
also judges it, so lean on the deterministic script for anything numeric (it does
not rely on the model's judgment), and keep the semantic pass strictly
adversarial. Report uncertainty honestly rather than rubber-stamping.
