# Simple mode + qwen3.8: two near-zero-score papers — analysis and status

**Status: both papers diagnosed and manually fixed (re-run, re-scored). The general
validity-check-plus-retry mechanism that would catch this class of bug automatically has been
proposed but not implemented — see "Proposed general fix" below.**

Found while reviewing the pk-summary benchmark's complete 18-combo matrix
(`benchmark_results/20261007-baseline-fix/` on scratch): `simple_qwen38` scored far below every
other combo on 2 of its 9 papers.

| PMID | before | after | what changed |
|---|---|---|---|
| 30825333 | semantic 20.0, F1 strict 0.000 | semantic 80.0, F1 strict 1.000 | re-run; row alignment now correct |
| 34183327 | semantic 0.0, F1 strict 0.000 | semantic 88.0, F1 strict 1.000 | re-run; its real PK table now extracts all 30 rows |

Fixing these 2 papers took `simple_qwen38`'s overall score from 74.3/0.710 (worst combo in the
18-combo matrix) to 90.8/0.932 (2nd-best, behind only `simple_gpt54`).

## Issue 1: a silently-dropped field shifts every later column (PMID 30825333)

### Symptom

The model's raw output for this paper's table_3:

```
Ketamine,Ketamine,,children,,Total area under curve,91,Mg/min/L,mean,SD,30,,,,,
```

against the 16-column schema (`Drug name,Analyte,Specimen,Population,Pregnancy
stage,Subject N,Parameter type,Value,Unit,Summary statistics,Variation type,Variation
value,Interval type,Lower limit,High limit,P value`). Position 6 (Subject N) holds
`"Total area under curve"` — a parameter name, not a number — and every field after it is
similarly one position off: position 7 (Parameter type) holds `91`, position 8 (Value) holds
`Mg/min/L`, and so on. Scored against the baseline this way, the row fails on essentially every
column, hence the 0.000 strict F1.

### Root cause

The model never wrote a value for "Subject N" (the baseline has `19` for every row), but also
never wrote its placeholder comma — so the next real value (`Total area under curve`) slid into
Subject N's slot, and everything downstream followed. The **total field count still came out
correct** (16 fields, matching the schema) because the model padded the row with enough trailing
empty fields to compensate. `simple_qwen38`'s prompt (`prompts/simple_prompts/pk_summary.md`)
explicitly warns against exactly this failure mode:

> Every row must have exactly 15 commas. ... Leave one out and every later column shifts by one,
> which makes the whole row wrong. Count the commas before you write the row.

qwen3.8 violated this instruction on this specific row anyway. "Simple" mode is deliberately the
unguarded baseline arm (`scripts/simple_prompt/run_simple_prompt.py`'s own docstring: "no
decomposition, no verification"), so nothing caught it.

### Why a naive "is the column count correct" check would not have caught this

The field count was already correct (16/16) — the bug is a *semantic* misalignment, not a
*count* mismatch. A validity check that only counts commas/fields would pass this row
unchanged. The codebase already has the right primitive for the smarter check, though:
`scripts/simple_prompt/_common_summary.py`'s `_realign()` scores every candidate
insert/drop-position combination using `_fits(column_name, cell)` — a per-column type-plausibility
heuristic (numeric columns must look numeric, vocabulary columns must match known values, text
columns must not be bare numbers) — and keeps whichever arrangement scores best. **This is
currently only invoked when `len(row) != len(header)`** (`_common_summary.py:200`); a
count-correct-but-shifted row like this one never reaches it. Extending validity-checking to also
run (or re-run) that same type-plausibility score on count-correct rows — flagging a row as
invalid when the as-given alignment scores clearly worse than some shifted variant would — closes
this gap without inventing new logic.

### Fix applied

Manual, scoped re-run: moved the paper's stale `results/simple_qwen38/30825333/` output aside,
then re-ran `run_simple_prompt.py --pmids 30825333,34183327` (see Issue 2) with the same prompt,
model, and settings — no code or prompt change. The retry produced a correctly-aligned row
(Subject N and Specimen now correctly blank instead of holding shifted values). Not a guaranteed
fix in general (a genuinely deterministic model should reproduce the same mistake), but it worked
here, plausibly due to GPU-level nondeterminism in kernel execution order even at temperature 0 —
consistent with the same "bare retry helps anyway" pattern documented for the skills-mode qwen3.8
retries in `docs/SKILLS_QWEN38_TIME_CONSUMING_ISSUE.md`.

## Issue 2: zero rows on the paper's actual PK data table (PMID 34183327)

### Symptom

All 3 of this paper's tables produced zero rows:

- **table_1** — "Baseline characteristics and drug doses of Indonesian children with TBM"
  (demographics: age, sex, GCS, CSF values, drug doses). **Correctly** 0 rows — the prompt
  explicitly excludes demographics-only tables with no PK parameter.
- **table_2** — "Summary of pharmacokinetic (PK) parameters of isoniazid, rifampicin and
  pyrazinamide among Indonesian children treated for TBM" — AUC/Cmax/CSF-concentration values for
  3 drugs across 2 assessment time points. **This is the paper's actual target data** (the
  baseline's 30 rows come from here) and should not have produced 0 rows.
  **This is the bug.**
- **table_3** — a multivariate linear regression table (coefficients, 95% CIs, R²).
  **Correctly** 0 rows — the prompt excludes "regression models, covariate analyses."

### Root cause

table_2's layout nests each drug's parameter rows under a blank drug-name "subheader" row
(`Isoniazid` / `Rifampicin` / `Pyrazinamide` as group labels with no data, followed by their own
indented `AUC0–24` / `Cmax` / `CCSF0–2` / ... rows). The prompt gives detailed, explicit guidance
for one other non-trivial layout (the transposed N/Range/Mean±SD/Median table), but has no
equivalent walkthrough for this drug-as-subheader nesting. qwen3.8 appears to have given up on the
whole table rather than attempting a partial extraction.

### Fix applied

Same scoped re-run as Issue 1 (`--pmids 30825333,34183327`). table_2 produced all 30 rows
correctly on retry — drug name, specimen (plasma/CSF), Subject N (20 and 12, matching the two
assessment groups), and all AUC/Cmax/CCSF parameter values and intervals matched the baseline
exactly (strict F1 1.000).

### Caveat for a future "retry if empty" check

A blanket "table produced zero rows → invalid → retry" rule has a real false-positive cost: empty
output is the *correct* answer for demographics-only and regression-only tables (as table_1 and
table_3 above both correctly did), and the prompt says so explicitly ("If no row in the table
qualifies, output the header row and nothing else"). A naive empty-check would force a wasted
duplicate call on every such legitimately-empty table. Given "simple" mode's low per-call cost and
a 1-retry cap, this is a bounded, acceptable cost — not a reason to skip the check — but worth
knowing so a retry triggered on a legitimately-empty table isn't mistaken for the checker
malfunctioning.

## Side-effect bug found while manually fixing this: `--pmids` is not merge-safe

While re-running the two papers above, `run_simple_prompt.py --pmids 30825333,34183327` was used
to avoid re-processing the other 7 (already-correct) papers. This surfaced a separate, smaller
bug: the script's manifest write

```python
json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
```

(`run_simple_prompt.py:296`) always opens `manifest.json` in `"w"` mode and dumps only the
current run's `manifest` dict — built solely from `pmids`, the list this invocation processed.
It never reads back the existing file first, so a run scoped to a subset of papers **silently
discards every other paper's entry** from `manifest.json` (and, by the same mechanism,
`all_predictions.csv`). The per-paper `combined.csv` files themselves are unaffected (each is
written to its own `pdir`, independent of the other papers) — this only loses the
pooled/aggregate bookkeeping files.

**Workaround applied:** recovered accurate per-paper elapsed time for the 7 affected papers from
the original full run's job log (`err_and_out/simple_qwen38-7720671.out` on scratch, which prints
each table's elapsed time even though it doesn't echo token counts); token counts for those 7
papers are unrecoverable and are left blank in `time_tokens.csv` with a note, rather than guessed.
Scoring was unaffected throughout, since the benchmark's scorer reads each paper's `combined.csv`
directly, not the pooled files.

**Suggested fix (not implemented):** before writing, load the existing `manifest.json` /
`all_predictions.csv` (if present) and merge in only the entries for `pmids` actually processed
this run, instead of overwriting wholesale. Anyone re-running `run_simple_prompt.py --pmids ...`
against an `--out` directory that already has other papers' results should back up
`manifest.json` and `all_predictions.csv` first until this is fixed.

## Proposed general fix (discussed, not yet implemented)

Add a validity check to `run_simple_prompt.py`'s per-table loop, after parsing a table's response
and before accepting it, with **one bare retry** (same prompt, no feedback) if invalid:

- **Invalid if:** any data row's field count doesn't match the schema AND can't be fixed by
  `_realign`, **or** (new) the as-given count-correct alignment scores clearly worse under
  `_fits()` than some shifted variant (catches Issue 1's class of bug), **or** the table produced
  zero rows (catches Issue 2's class of bug, with the empty-is-sometimes-correct caveat above).
- **Retry cap of 1**, matching the pattern already established for `pk_individual`/`pk_summary`'s
  `RetryException` guards elsewhere in this codebase, but without the retry-with-feedback
  machinery those use (simple mode is intentionally a single bare LLM call, not an agent loop) —
  just "ask once more with the identical prompt."
- Log which condition triggered a retry and whether it resolved, in the same per-table verbose
  style `run_simple_prompt.py` already uses (`table_X: N rows (Ts, ...)`).

## Related files

- `scripts/simple_prompt/run_simple_prompt.py` — the simple-mode runner; where the proposed
  retry mechanism and the `--pmids` manifest-merge fix would both go
- `scripts/simple_prompt/_common_summary.py` — `_realign`/`_fits` (the existing type-plausibility
  scorer to extend), `parse_csv` (where `_realign` is currently gated on count mismatch only)
- `prompts/simple_prompts/pk_summary.md` — the prompt; already warns about the comma-count issue
- `benchmark_results/20261007-baseline-fix/` (on scratch) — the re-scored pk-summary matrix
  reflecting both fixes; `scores_summary.csv`'s `simple_qwen38` row description notes the re-run
- `docs/SKILLS_QWEN38_TIME_CONSUMING_ISSUE.md` — the other qwen3.8 investigation this session,
  including the same "a bare retry helped despite temperature 0" observation for skills mode
