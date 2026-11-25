# CurateAgent Cascade — Live Tier 1+2+3 Calibration Results

Live runs of `ValidityChecker.check_validity()` (Tier 1 structural + Tier 2 existence +
Tier 3 LLM verification, see `extractor/curate_agent/validity_checker.py`) against Arm A's
real curated CSVs, via `scripts/calibrate_validity_threshold.py --live`, submitted as one
Slurm job per PMID. Tier 3 used `AGENT_LLM` matching the dataset's own Arm A model in each
case (`QWEN3.8-27B` for pk_individual, `QWEN3.6-35B` for pk_summary), served from a local
Ollama instance on the job's own GPU node.

Columns: `error_A` is `ValidityChecker`'s combined error rate on Arm A's curated table;
`tier1(d/n)` is dropped-rows/total-rows; `tier3(f/n)` is
`(error_row_count + missed_row_count)/(total_rows + missed_row_count)`'s numerator/denominator;
`F1_A`/`F1_B` are Arm A's (simple one-shot) and Arm B's (multi-agent) strict F1 against gold,
from the benchmark harness's own `result_f1.log`; `gap` is `F1_B - F1_A`.

Escalation decision: `escalate` if `error_A > threshold` (default `0.2`). An escalation is
"worth it" when `gap > epsilon` (`0.05`).

---

## pk_individual (`pk_individual_qwen38` dataset)

Arm A: `benchmark/data/pk-individual/2026-10-01-collect-simple_qwen38/` (qwen3.8, one-shot).
Arm B: `benchmark/data/pk-individual/2026-10-01-collect-multiagent_qwen38/` (qwen3.8, multi-agent).
8 PMIDs (a 9th, `33253437`, has no multi-agent output and is excluded).

Run with the **improved** Tier 3 prompt (`CURATE_AGENT_VERIFICATION_PROMPT` in
`extractor/curate_agent/llm_verification.py`), which explicitly checks wide-source-table
completeness column-by-column and treats negative/null findings ("Not measured", "Not
detected", etc.) as real data points whose omission counts as a missing row. An earlier run
with the prompt before this fix found zero issues on every single paper.

```
pmid        error_A  tier1(d/n) tier3(f/n)   F1_A   F1_B    gap
10971311      0.000        0/98       0/98  0.714  0.990 +0.275
11849190      0.000        0/83       0/83  0.976  1.000 +0.024
18426260      0.017        0/94       5/99  1.000  0.363 -0.637
23200982      0.025       0/134     11/145  0.893  0.921 +0.027
24989434      0.036        0/84       0/84  1.000  0.636 -0.364
32056930      0.000       0/170      0/170  0.682  0.886 +0.204
32153014      0.025        0/25       2/27  0.909  1.000 +0.091
34746508      0.214         0/5       9/14  0.588  0.828 +0.239
```

Classification at the default `threshold=0.2`, `epsilon=0.05`:

| pmid | error_A | gap | outcome |
|---|---|---|---|
| 10971311 | 0.000 | +0.275 | missed_escalation |
| 11849190 | 0.000 | +0.024 | stayed_correctly (gap below epsilon) |
| 18426260 | 0.017 | -0.637 | stayed_correctly (A genuinely better) |
| 23200982 | 0.025 | +0.027 | stayed_correctly (gap below epsilon) |
| 24989434 | 0.036 | -0.364 | stayed_correctly |
| 32056930 | 0.000 | +0.204 | missed_escalation |
| 32153014 | 0.025 | +0.091 | missed_escalation |
| **34746508** | **0.214** | **+0.239** | **escalated_correctly** |

**1 escalated_correctly, 3 missed_escalation, 4 stayed_correctly, 0 wasted.**

`34746508`'s escalation was triggered by a genuine completeness gap: the source table
explicitly listed "Not measured"/"Not detected" results for several cases, and Arm A silently
dropped all of them, keeping only rows with a numeric value. The improved prompt caught 9 of
the 10 actual null-result cells.

The 3 `missed_escalation`s (`10971311`, `32056930`, `32153014`) were investigated in depth for
`32056930`: the real F1 evaluator's own row-matching logic shows the underlying data (Patient
ID, Analyte, Parameter value) is correct in nearly every "miss" - what drags those rows below
the strict F1 threshold is Arm A using *generic* descriptive labels (e.g. `"concentration-dose
ratio"`) where gold uses the paper's own *specific* phrasing (e.g. `"Maternal serum - C/D AM"`).
Neither our Tier 3 prompt nor the unmodified multi-agent mode's own verification prompt flags
this - both explicitly instruct the model to treat "slight wording differences" as acceptable,
so this class of F1 gap (label genericity, not factual error) is a known, deliberate blind spot
of the current three tiers, confirmed against both prompt variants on live data.

---

## pk_summary (`pk_summary_qwen36` dataset)

Arm A: `benchmark/data/pk-summary/2026-9-25-collect-simple_qwen36/` (qwen3.6, one-shot).
Arm B: `benchmark/data/pk-summary/2026-9-25-collect-multiagent_gemma4/` (gemma4, multi-agent).
9 PMIDs.

```
pmid        error_A  tier1(d/n) tier3(f/n)   F1_A   F1_B    gap
16143486      0.203        2/32      35/64  0.762  0.984 +0.222
17635501      0.049        2/15       0/15  0.867  0.897 +0.030
22050870      0.146        0/42      29/66  0.857  0.988 +0.131
29943508      0.333         0/6        6/6  0.667  0.800 +0.133
30825333      0.000         0/8        0/8  0.941  0.941 +0.000
34114632      0.133        0/30      12/30  0.750  0.750 +0.000
34183327      0.011        0/30       1/30  0.500  0.500 +0.000
35465728      0.185         0/8      10/18  0.667  0.250 -0.417
35489632      0.083         0/8        2/8  1.000  1.000 +0.000
```

Classification at the default `threshold=0.2`, `epsilon=0.05`:

| pmid | error_A | gap | outcome |
|---|---|---|---|
| **16143486** | **0.203** | **+0.222** | **escalated_correctly** |
| 17635501 | 0.049 | +0.030 | stayed_correctly |
| 22050870 | 0.146 | +0.131 | missed_escalation (close - just under threshold) |
| **29943508** | **0.333** | **+0.133** | **escalated_correctly** |
| 30825333 | 0.000 | +0.000 | stayed_correctly |
| 34114632 | 0.133 | +0.000 | stayed_correctly |
| 34183327 | 0.011 | +0.000 | stayed_correctly |
| 35465728 | 0.185 | -0.417 | stayed_correctly (A genuinely better) |
| 35489632 | 0.083 | +0.000 | stayed_correctly |

**2 escalated_correctly, 1 missed_escalation, 6 stayed_correctly, 0 wasted.** Meaningfully
stronger signal than pk_individual on this sample - Tier 3 found real, substantial error counts
on several papers (`35/64`, `29/66`, `6/6`, `12/30`, `10/18`), and those elevated error rates
line up with real F1 gaps far more often here. `22050870` is a near-miss and would also be
caught by a slightly lower threshold (e.g. `0.15`), at the cost of some wasted-escalation risk
elsewhere - not yet swept.

**Caveat on `30825333`'s `F1_A`/`F1_B`:** these come from `result_f1.log`, which was scored
against an older gold file (`30825333_baseline_original.csv`, 9 rows) containing a row since
removed from the corrected gold (`30825333_baseline.csv`, 8 rows, used by this calibration run
for Tier 2/3's own source lookups). A fresh F1 computation against the corrected 8-row gold
gives `F1_A = 1.0`, not `0.941` - the `gap = +0.000` conclusion for this paper doesn't change,
but the absolute `F1_A`/`F1_B` values quoted above for this one PMID are stale relative to the
current gold standard.

---

## v2 per-table cascade — first live round (2026-10-30)

`scripts/run_cascade_benchmark.py`, one Slurm task per paper (`scripts/slurm/slurm_cascade_v2_*_array.sh`),
route skipped (`schemas=` fixed), Arm A = simple prompt on the dataset's own model, Arm B =
execution-only multi-agent workflow, Tier 3 + judge on `AGENT_LLM`; one model per dataset
(pk_summary: qwen3.6:35b / QWEN3.6-35B, so F1_B below is the **multiagent_qwen36** collection;
pk_individual: qwen3.8:27b / QWEN3.8-27B, F1_B = multiagent_qwen38). F1 = strict, cascade's
`combined.csv` scored live against gold with the benchmark's `TablesF1Evaluator`; F1_A/F1_B from
the historical `result_f1.log`s. `sel` = tables the selector kept, `esc` = tables escalated.

### pk_summary (`pk_summary_qwen36`)

| pmid | sel | esc | F1_cascade | F1_A | F1_B | vs A | vs B | LLM calls | sec |
|---|---|---|---|---|---|---|---|---|---|
| **16143486** | 3 | 2 | **0.984** | 0.762 | 0.853 | +0.222 | +0.131 | 9 | 531 |
| 17635501 | 1 | – | 0.897 | 0.867 | 0.849 | +0.030 | +0.048 | 3 | 707 |
| 22050870 | 2 | – | 0.690 | 0.857 | 0.407 | −0.167 | +0.283 | 5 | 942 |
| 29943508 | 1 | 1 | 1.000 | 0.667 | 0.667 | +0.333 | +0.333 | 5 | 235 |
| 30825333 | 1 | – | 1.000 | 0.941 | 0.941 | +0.059 | +0.059 | 3 | 89 |
| 34114632 | 2 | – | 1.000 | 0.750 | 0.462 | +0.250 | +0.538 | 5 | 938 |
| 34183327 | 1 | – | 1.000 | 0.500 | 0.571 | +0.500 | +0.429 | 3 | 541 |
| 35465728 | 1 | 2 | 0.000 | 0.667 | 0.000 | −0.667 | +0.000 | 5 | 314 |
| 35489632 | 1 | – | 1.000 | 1.000 | 0.556 | +0.000 | +0.444 | 3 | 304 |
| **mean** | | | **0.841** | 0.779 | 0.589 | | | 4.6 | 511 |

Beats or ties Arm A on 7/9, Arm B on 9/9; 3 papers escalated (one table each).
16143486 — the paper that motivated v2 — now escalates only table 2 (Tier 1 catches 16/18 shifted
rows), the judge picks B, and Tier 3 with the summary-level completeness rules reports 0 missing
rows on table 4 (v1 reported 32). 9 LLM calls vs v1's ≈48–75.

### pk_individual (`pk_individual_qwen38`)

| pmid | sel | esc | F1_cascade | F1_A | F1_B | vs A | vs B | LLM calls | sec |
|---|---|---|---|---|---|---|---|---|---|
| 10971311 | 2 | – | 0.714 | 0.714 | 0.990 | 0.000 | −0.276 | 5 | 333 |
| 11849190 | 2 | – | 0.747 | 0.976 | 1.000 | −0.229 | −0.253 | 5 | 317 |
| 18426260 | 2 | 2 | 0.793 | 1.000 | 0.363 | −0.207 | +0.430 | 6 | 560 |
| 23200982 | 3 | – | 0.893 | 0.893 | 0.921 | 0.000 | −0.027 | 7 | 635 |
| 24989434 | 1 | – | 1.000 | 1.000 | 0.636 | 0.000 | +0.364 | 3 | 298 |
| 32056930 | 2 | – | 0.682 | 0.682 | 0.886 | 0.000 | −0.204 | 5 | 543 |
| 32153014 | 1 | – | 0.909 | 0.909 | 1.000 | 0.000 | −0.091 | 3 | 249 |
| 34746508 | 2 | 2,3 | 0.526 | 0.588 | 0.828 | −0.062 | −0.301 | 7 | 309 |
| **mean** | | | 0.783 | 0.845 | 0.828 | | | 5.1 | 406 |

### What this round showed (and what is already fixed)

1. **Source-table rendering defect (fixed in `curate_orchestrator._flatten_header`).** Tables
   with multi-row HTML headers come out of `pmid_info.db` with their MultiIndex tuples stringified
   (`('Unnamed: 0_level_0', 'Patient') | ('Venlafaxine', 'Cmax1(µg l−1)')`), and
   `write_table_markdown` fed that to Arm A verbatim. The historical simple-mode runs never saw it
   (they read the skill converter's flattened markdown). **17 source tables in 11 of the 17 papers**
   were affected — 11849190 (both tables: 12/48 rows got an empty `Parameter type`), 35465728
   (table 2: every column shifted → F1 0.000), 10971311, 32056930, 32153014, 23200982, 18426260,
   24989434, 34114632, 35489632, 22050870. The pk_individual numbers above are therefore a lower
   bound; headers are now flattened (`" - "`-joined, `Unnamed:` levels dropped) before rendering.
2. **The 0.2 gate is too lax per table** (plan item 6). Genuinely flawed tables land at
   0.08–0.14 because `error_rate` averages three tier rates and a clean Tier 2 dilutes real Tier 1/3
   hits: 22050870 table 2 (two column-shifted rows: Tier 1 1/12, Tier 3 2 rows → 0.083), table 3
   (Tier 3: 2 wrong cells + 18 missing rows → 0.139). To be re-picked with
   `calibrate_validity_threshold.py --live --per-table`; a max-of-tiers gate is the alternative.
3. **Run-to-run variance of the arms is large.** 18426260 table 2: fresh Arm A returned a
   header-only CSV in 8 s on a plain-header table (historical run: F1 1.000). The cascade handled it
   correctly (0 rows → escalate → Arm B → judge → 0.793 instead of losing the table), but
   "cascade vs historical Arm A" conflates this variance with the design change.
4. **Execution-only Arm B can return an empty table** without raising (34746508 table 2, 0 rows
   from an 11-row `Case 1…Case 5` table). The judge kept Arm A both times this happened.
5. **Tier 3 recall on pk_individual is the remaining gate weakness**: 10971311, 32056930,
   32153014, 23200982 all report `error_A ≈ 0` on every table while Arm B would have scored
   +0.03 to +0.28 higher. Per-table scoping did not change that; it is a prompt/recall problem.
6. **Ops**: qwen3.6 on `ollama.sif` crashed 4 of 9 pk_summary tasks (CUDA illegal memory access,
   the documented `ollama/ollama#17434`); they were re-run on `ollama-old.sif`, which the Slurm
   script now pins. Mean wall-clock per paper 406–511 s, dominated by Arm A's single call
   (70–470 s on the 35B thinking model).

## Caveats (apply to both tables)

- Small samples (8-9 papers each) - a first-pass sanity check, not a statistically solid number.
- `error_A` has no `extra_dropped_rows` input here (no `manifest.json` for these historical
  benchmark runs), so Tier 1's structural count is a lower bound, not the full picture a live
  Arm A run with a real manifest would give.
- Arm B's own CSV is never scored by `ValidityChecker` in this exercise - only its
  already-computed F1 is used as the ground-truth signal for "would escalating have helped."
- Tier 3 is a single LLM pass (no correction loop); see
  `extractor/curate_agent/llm_verification.py` for the deterministic guards (no-op/out-of-range
  idx filtering) applied to its raw output before counting errors.

---

## Tier 4 audit of the first live round (2026-10-31)

For each of the six escalated tables, the paper's `combined.csv` was re-built twice - once with
Arm A's table, once with Arm B's - and scored against gold. The judge's pick is right when it
gave the higher paper F1. All six verdicts came from the reports-only pass (`used_full_tables`
false, ~4 s each); the judge never saw a table, and every pick equals "lower error_rate wins".

| pmid / table | rows A / B | error_rate A / B | judge | F1 if A | F1 if B | right? |
|---|---|---|---|---|---|---|
| 16143486 t2 (pk_summary) | 18 / 16 | 0.333 / 0.219 | B | 0.444 | **0.984** | yes |
| 29943508 t1 (pk_summary) | 6 / 7 | 0.278 / 0.500 | A | **1.000** | 0.923 | yes |
| 35465728 t2 (pk_summary) | 8 / 28 | 0.405 / 0.500 | A | 0.000 | **0.444** | **no** |
| 18426260 t2 (pk_individual) | 0 / 49 | 1.000 / 0.176 | B | **1.000** | 0.793 | **no** |
| 34746508 t2 (pk_individual) | 5 / 0 | 0.222 / 1.000 | A | **0.526** | 0.000 | yes |
| 34746508 t3 (pk_individual) | 0 / 2 | 1.000 / 0.567 | B | **0.588** | 0.526 | **no** |

The three wrong picks, and what fixed each (details in `VERIFICATION_AGENT_DESIGN.md`, "Fixes
from the first live round's Tier 4 audit"):

- **35465728 t2** - Arm A's 8 rows had the right numbers one column left of home (the model
  dropped the empty `Pregnancy stage` cell and added an empty after `median`; field count still
  16). Tier 3 saw 0 errors; Tier 3 on Arm B counted one repeated `Adult → Neonate` label 28
  times. Fixed at parse time: `_reshift` moves the empty back. Re-parsed, the table scores
  **1.000**.
- **18426260 t2** - the source is "Mean ± SD (Range) of Drug Level-to-Dose Ratios", no patient
  IDs; gold has no rows from it. Arm A correctly returned nothing; the judge read `error_rate
  1.000, zero rows` as failure and took Arm B's 49 wrong rows. Fixed by the empty-table
  assessment in the gate, the judge rule that zero rows is correct when the source has no
  curatable data, and a pk_individual selector that excludes means-only tables.
- **34746508 t3** - a literature table of cases from other papers; gold does not curate them.
  Same mechanics and same fixes as above; the "other published studies" exclusion is annotation
  policy and is written into both the assessment and the selector prompts.

### Parse-time shift repair: regression over every saved Arm A reply

`_reshift` was run over all 59 saved `arm_a.raw.txt` files (the first live round's local and
scratch copies plus the calibration runs) and the output diffed against the CSV the old parser
wrote. 5 distinct tables changed; each was re-scored as the paper's combined table with the
re-parsed Arm A table swapped in (script and full diff:
`/fs/scratch/PCON0100/feng1426/projects/tabular-data-jobs/out/curate_agent_cascade/reparse_regression_20261031/`).

| pmid / table | rows changed | paper F1, first-round combined → with re-parsed Arm A table | what was shifted |
|---|---|---|---|
| 35465728 t2 (pk_summary) | 8 / 8 | 0.000 → **1.000** | the target case: `60 / Initial Cavg / 12.9 / mg/L / median` one column left |
| 11849190 t1 (pk_individual) | 12 / 48 | 0.747 → **0.892** | `maternal/pediatric` in `Pregnancy stage`, `M/PAUC` in `Parameter unit`, value in `Time value` |
| 17635501 t3 (pk_summary) | 1 / 14 | 0.897 → **0.966** | `6 / Bioavailability / 89.4 / %` one column right |
| 22050870 t2 (pk_summary) | 2 / 12 | 0.690 → **0.714** | `15 / AUC0−∞ / 822.5` one column left |
| 16143486 t2 (pk_summary) | 16 / 18 | 0.984 (Arm B's table) → 0.952 (repaired Arm A) | two segments per row, one dropped comma each; full repair needs two moves + the interval rule |

Two near-misses during development, both caught by this regression and fixed before the code
shipped: "3 Month to < 3 Years" in `Specimen` (22050870 t3) and "95% CI" in `Lower limit` were
read as numbers by `_fits`' front-anchored regex, which triggered a needless move on the first and
let a half repair of 16143486 pass as clean (`_fits_strict`); and with two moves allowed the
search parked "parturients" in free-text `Specimen` to dodge the −1 vocabulary miss in
`Population` (vocabulary misses now vote 0 inside the search).

Note on 16143486: the repaired Arm A table is 0.032 below Arm B's. If the gate now passes Arm A
(Tier 1 no longer flags the rows), the paper drops from 0.984 to 0.952; the first round's 0.984
depended on the shift being *left* unrepaired so that Tier 1 would escalate. Net over the five
tables the repair is clearly positive; the second live round shows what the gate does.

Two more defects surfaced in an interim second-round run (jobs 8230734/8230772, kept on scratch)
and were fixed before the final round below:

- **`_realign` dropped short rows on a free-text tie.** The fresh 35465728 reply had 4 of 8 rows
  one field short; inserting the empty into the existing `,,,` run or between the two
  "Vancomycin" cells scores the same (Analyte/Specimen do not vote), and a tie meant "drop".
  22050870 table 3 lost 36 of 48 rows the same way. Ties are now broken toward extending an
  existing run of empties - the slip the repair exists for. Re-run over all 85 saved replies
  (first round + interim round): the 35465728 table goes 4 → 8 rows (paper 0.444 → 1.000 with
  Arm A), 22050870 table 3 goes 12 → 48 rows (0.588 → 0.588), nothing else changes.
- **The reports-only judge picked an empty Arm B over a 12-row Arm A** (22050870 table 3: B
  returned nothing, the assessment said the table has data, and the judge read that text as a
  point for B; paper 0.690 → 0.222). An empty candidate whose assessment says the source has
  curatable data now loses by rule to a candidate with rows, no LLM call
  (`curate_orchestrator._rule_verdict`); a *justified* empty B against an A with rows goes
  straight to the full-table judge pass.
- **Ops:** two jobs on one node shared Ollama's default port 11434; the second `ollama serve`
  failed to bind and silently used the first job's server (selector job 8230733 failed, the
  co-located pk_summary task saw 500s). Every Slurm script now runs Ollama on a per-job port.

## v2 per-table cascade — second live round (2026-10-31)

Same setup as the first round (route skipped, Arm A = simple prompt on the dataset's model, Arm B
= execution-only multi-agent, Tier 3 + judge on `AGENT_LLM`), with the three audit fixes, the
`_realign` tie-break and the judge rule. pk_summary ran entirely on `ollama-old.sif` (v0.20.0;
the first round mixed images after the qwen3.6 crash). `LLM calls` still excludes Arm B's own
workflow calls (follow-up). Jobs 8230894 (pk_summary) and 8230896 (pk_individual); work dirs
`.curate_agent_calibration/<dataset>_r2/`, synced to scratch under `cascade_v2r2_*`.

### pk_summary (`pk_summary_qwen36`)

| pmid | sel | esc | F1_cascade | round 1 | F1_A | F1_B | vs A | vs B | LLM calls | sec |
|---|---|---|---|---|---|---|---|---|---|---|
| 16143486 | 3 | – | 0.952 | 0.984 | 0.762 | 0.853 | +0.190 | +0.099 | 7 | 434 |
| 17635501 | 1 | – | 0.867 | 0.897 | 0.867 | 0.849 | +0.000 | +0.018 | 3 | 218 |
| 22050870 | 2 | – | 0.588 | 0.690 | 0.857 | 0.407 | −0.269 | +0.181 | 5 | 344 |
| 29943508 | 1 | 1 | 1.000 | 1.000 | 0.667 | 0.667 | +0.333 | +0.333 | 5 | 226 |
| 30825333 | 1 | – | 1.000 | 1.000 | 0.941 | 0.941 | +0.059 | +0.059 | 3 | 90 |
| 34114632 | 2 | – | 0.750 | 1.000 | 0.750 | 0.462 | +0.000 | +0.288 | 5 | 285 |
| 34183327 | 1 | – | 1.000 | 1.000 | 0.500 | 0.571 | +0.500 | +0.429 | 3 | 185 |
| **35465728** | 1 | 2 | **1.000** | 0.000 | 0.667 | 0.000 | +0.333 | +1.000 | 5 | 264 |
| 35489632 | 1 | – | 1.000 | 1.000 | 1.000 | 0.556 | +0.000 | +0.444 | 3 | 163 |
| **mean** | | | **0.906** | 0.841 | 0.779 | 0.589 | | | 4.3 | 245 |

Beats or ties Arm A on 8/9 (first round 7/9), Arm B on 9/9. Two escalations, both judged for
Arm A, both right (29943508: A 1.000 vs B 0.923; 35465728: A 1.000 vs B 0.444). No pk_summary
table was empty, so the empty-table assessment did not fire here.

- **35465728** is the audit case: Arm A's rows are parsed in the right columns (this run's reply
  was short by one field per row - the `_realign` tie-break case, not the `_reshift` case), Tier
  3 still escalated on the 20 count rows it considers missing, and the judge picked A.
- **16143486**: as predicted, the repaired Arm A table passes the gate (Tier 1 clean, Tier 3 2
  error rows) and the paper lands at 0.952 instead of Arm B's 0.984.
- **Arm A is not reproducible run to run, and that is the biggest remaining source of variance.**
  Temperature 0, same prompt, same model, same image: 22050870 table 3 came back with 30 rows in
  the first round, 48 in the interim run (36 of them one field short) and 48 here; 34114632
  table 2 came back with 24 rows in the first round and 12 rows here (the 12 rows
  without a P value were left out; Tier 3 reported 0 missing, so the gate did not escalate);
  17635501 wrote "t1/2 (elimination)" once and "t1/2" the next time. The 22050870 and 34114632
  drops against the first round are this, not the fixes; 22050870 table 2 also changed from a
  shifted layout (repaired by `_reshift` in the regression) to a correct one.

### pk_individual (`pk_individual_qwen38`)

| pmid | sel | esc | F1_cascade | round 1 | F1_A | F1_B | vs A | vs B | LLM calls | sec |
|---|---|---|---|---|---|---|---|---|---|---|
| 10971311 | 2 | – | 0.714 | 0.714 | 0.714 | 0.990 | 0.000 | −0.276 | 5 | 366 |
| 11849190 | 2 | – | 0.976 | 0.747 | 0.976 | 1.000 | 0.000 | −0.024 | 5 | 378 |
| **18426260** | 1 | – | **1.000** | 0.793 | 1.000 | 0.363 | 0.000 | +0.637 | 3 | 288 |
| 23200982 | 3 | – | 0.893 | 0.893 | 0.893 | 0.921 | 0.000 | −0.028 | 7 | 598 |
| 24989434 | 1 | – | 1.000 | 1.000 | 1.000 | 0.636 | 0.000 | +0.364 | 3 | 254 |
| 32056930 | 2 | – | 0.841 | 0.682 | 0.682 | 0.886 | +0.159 | −0.045 | 5 | 470 |
| 32153014 | 1 | – | 0.667 | 0.909 | 0.909 | 1.000 | −0.242 | −0.333 | 3 | 104 |
| **34746508** | 1 | 2 | **0.588** | 0.526 | 0.588 | 0.828 | 0.000 | −0.240 | 4 | 120 |
| **mean** | | | **0.835** | 0.783 | 0.845 | 0.828 | | | 4.4 | 322 |

Both audit targets land where the counterfactual said: **18426260** → 1.000 (the new selector
dropped the means-only table 2, so only table 1 was curated: 1 table, 3 calls, no escalation)
and **34746508** → 0.588 (table 3, the literature table, dropped by the selector; on table 2 Arm
B again returned nothing, the assessment said the table has data, and the rule kept Arm A's 5
rows with no judge call). Selection matched gold's source tables on every paper (selector check
job 8230773).

- **11849190** 0.747 → 0.976: table 1's 12 shifted rows were repaired by `_reshift` in the
  regression, and this run's reply had no shift at all - either way the 48 rows now land.
- **32056930** 0.682 → 0.841 and **32153014** 0.909 → 0.667 are Arm A run-to-run variance, as in
  pk_summary. On 32153014 Tier 3 did see 13 missing rows against 15 curated, but the three-tier
  average (0.155) stayed under the 0.2 gate - the "gate too lax per table" finding from the first
  round, still open.
- **Tier 3 recall on pk_individual is still the limiting factor**: 10971311 (Arm B +0.276),
  11849190 and 23200982 report 0 errors and ≤2 missed rows on every table, so nothing escalates
  and the cascade cannot do better than Arm A there.

### Round 2 against the success bar

| bar | result |
|---|---|
| 35465728 ≥ 0.667, no escalation | 1.000; escalated (Tier 3 counts the 20 count rows as missing) but the judge picked A |
| 18426260 = 1.000, no escalation, no judge | yes, yes, yes |
| 34746508 = 0.588, no escalation, no judge | 0.588; table 2 escalated (Arm A dropped 10 null rows) and was settled by rule, no judge call |
| no other paper drops vs round 1 | **not met**: 16143486 −0.032 (expected), 17635501 −0.030, 22050870 −0.102, 34114632 −0.250, 32153014 −0.242 - all Arm A run-to-run variance, none caused by the fixes |
| pk_individual mean ≥ Arm A's 0.845 | 0.835, just under; pk_summary mean 0.906 vs Arm A 0.779 |

Mean over both datasets: round 1 0.814 → round 2 0.873. LLM calls per paper 4.3–4.4 plus Arm B's
uncounted workflow calls on the 3 escalated tables.

## v2 per-table cascade — third live round, pk_individual with Arm B = gemma4 (2026-10-31)

Same code as round 2. Arm A unchanged (`qwen3.8:27b` simple prompt, re-run live); Arm B, Tier 3,
the empty-table assessment and the judge on `GEMMA4-31B` (`AGENT_LLM` = `PIPELINE_LLM`), so
F1_B is the multiagent_gemma4 collection, which covers all 9 gold papers - 33253437 is in scope
for the first time. Dataset `pk_individual_qwen38_gemma4`, job 8236465 (9 tasks, 7–15 min
each), work dir `.curate_agent_calibration/pk_individual_qwen38_gemma4/`, synced to scratch
under `cascade_v2r3_pki_gemma4-8236465_*`. Both models shared one GPU, so Ollama swapped them
between the Arm A call and the Tier 3 calls on every table (the longer per-paper times).

| pmid | sel | esc | F1_cascade | round 2 | F1_A | F1_B (gemma4) | vs A | vs B | LLM calls | sec |
|---|---|---|---|---|---|---|---|---|---|---|
| 10971311 | 2 | – | 0.714 | 0.714 | 0.714 | 0.821 | 0.000 | −0.107 | 5 | 536 |
| 11849190 | 2 | – | 0.976 | 0.976 | 0.976 | 0.772 | 0.000 | +0.204 | 5 | 542 |
| 18426260 | 1 | – | 1.000 | 1.000 | 1.000 | 0.757 | 0.000 | +0.243 | 3 | 485 |
| 23200982 | 3 | – | 0.893 | 0.893 | 0.893 | 0.883 | 0.000 | +0.010 | 7 | 803 |
| 24989434 | 1 | – | 1.000 | 1.000 | 1.000 | 0.667 | 0.000 | +0.333 | 3 | 376 |
| 32056930 | 2 | – | 0.841 | 0.841 | 0.682 | 0.876 | +0.159 | −0.035 | 5 | 705 |
| 32153014 | 1 | – | 0.667 | 0.667 | 0.909 | 1.000 | −0.242 | −0.333 | 3 | 418 |
| 33253437 | 1 | – | 0.912 | n/a | 0.912 | 0.784 | 0.000 | +0.128 | 4 | 585 |
| 34746508 | 1 | 2 | 0.588 | 0.588 | 0.588 | 0.649 | 0.000 | −0.061 | 4 | 396 |
| **mean** | | | **0.844** | 0.835 (8 papers) | 0.853 | 0.801 | | | 4.3 | 538 |

Over the 8 papers both rounds share, round 3 reproduced round 2 exactly (0.835 vs 0.835): every
table was won by Arm A, and the one escalation (34746508 table 2) was again settled by the
`unjustified_empty_loses` rule because gemma4's Arm B, like qwen3.8's, returned no rows while the
assessment listed 8 expected rows. **The cascade output on pk_individual is therefore identical
to the live Arm A output on all 9 papers**, and the LLM judge has still never run on this schema.

Per-table gate results (Arm A, `error_rate` = mean of the three tiers, gate 0.2):

| table | rows | Tier 2 unsupported | Tier 3 error rows | Tier 3 missed rows | error_rate | escalated |
|---|---|---|---|---|---|---|
| 10971311 t1 / t2 | 56 / 42 | 0 / 0 | 0 / 0 | 0 / 0 | 0.000 / 0.000 | – |
| 11849190 t1 / t2 | 48 / 35 | 0 / 0 | 0 / 0 | 0 / 0 | 0.000 / 0.000 | – |
| 18426260 t1 | 94 | 0 | 0 | 9 | 0.029 | – |
| 23200982 t2 / t3 / t4 | 40 / 41 / 53 | 0 / 0 / 0 | 0 / 0 / 0 | 4 / 2 / **34** | 0.030 / 0.016 / **0.130** | – |
| 24989434 t1 | 84 | 25 / 231 | 0 | 0 | 0.036 | – |
| 32056930 t2 / t3 | 128 / 42 | 0 / 0 | 0 / 0 | 12 / 12 | 0.029 / 0.074 | – |
| 32153014 t2 | 15 | 0 | 0 | **13** | **0.155** | – |
| 33253437 t2 | 141 | 0 | 0 | 11 | 0.024 | – |
| 34746508 t2 | 5 | 0 | 2 | 10 | 0.267 | yes → rule, A |

- **The gate, not the judge, is what limits pk_individual.** Tier 3 (gemma4 here, qwen3.8 in
  round 2) does report the problem on the two papers where Arm B would win: 32153014 (13 missed
  rows against 15 curated; gold has 30) and 23200982 table 4 (34 missed against 53). Averaging
  the Tier 3 rate with a clean Tier 1 and Tier 2 dilutes 0.46 and 0.39 to 0.155 and 0.130, both
  under 0.2. 10971311 (Arm B +0.107 with gemma4, +0.276 with qwen3.8) is the other kind: Tier 3
  reports 0 errors and 0 missed rows on both tables, so no gate setting would catch it.
- **Arm A live vs the collection**: 32153014 came back with 15 rows in both round 2 and round 3
  (the collection CSV has 25), so the 0.909 → 0.667 gap is a stable difference between the live
  re-run and the stored collection, not noise; 32056930 went the other way (+0.159) in both rounds.
- **Tier 4 evidence on pk_individual after three rounds**: 1 escalation per round, all three on
  34746508 table 2, all settled by the empty-table rule, 0 LLM judge calls. Evaluating the judge
  on this schema needs either a stricter gate (weight Tier 3's missed-row rate instead of
  averaging it away) or the offline A/B-pair benchmark from the Tier 4 plan.

### Time cost of round 3

Slurm task wall-clock from `sacct` (includes Ollama start-up, model loads and the final rsync);
cascade seconds from `cascade_report.json` (`seconds.total` and the per-stage sums over the paper's
tables). Total GPU time across the 9 tasks: 1 h 31 min (5473 s), one A100 each.

| pmid | tables | Slurm task wall-clock | cascade total (s) | select | Arm A | gate A (Tier 1-3) | Arm B | gate B | judge |
|---|---|---|---|---|---|---|---|---|---|
| 10971311 | 2 | 00:10:09 | 535 | 157 | 350 | 28 | – | – | – |
| 11849190 | 2 | 00:10:13 | 542 | 158 | 355 | 29 | – | – | – |
| 18426260 | 1 | 00:09:12 | 485 | 165 | 291 | 29 | – | – | – |
| 23200982 | 3 | 00:14:35 | 803 | 128 | 559 | 116 | – | – | – |
| 24989434 | 1 | 00:07:30 | 375 | 125 | 234 | 16 | – | – | – |
| 32056930 | 2 | 00:12:57 | 703 | 126 | 503 | 75 | – | – | – |
| 32153014 | 1 | 00:08:21 | 417 | 167 | 220 | 31 | – | – | – |
| 33253437 | 1 | 00:10:54 | 584 | 154 | 398 | 31 | – | – | – |
| 34746508 | 1 | 00:07:22 | 396 | 122 | 174 | 29 | 50 | 20 | – |
| **mean** | | | **538** | 145 | 343 | 43 | 6 | 2 | – |

Arm A (qwen3.8:27b simple prompt) is the dominant cost at about 64% of the cascade
time; the gate's one Tier 3 call per table is the next. Each stage boundary also pays a model swap
(qwen3.8 ↔ gemma4) since both share the GPU. Compared with round 2 (same 8 papers, one model, mean
322 s), the mean per-paper cascade time rose to 538 s.

## v2 per-table cascade — fourth live round, pk_individual, Arm B mode "pipeline" (2026-10-31)

Arm B now has an explicit mode switch (`arms.ARM_B_MODES`, `--arm-b-mode`,
`ARM_B_MODE=` in `scripts/slurm/slurm_cascade_v2_pki_qwen38_array.sh`): "pipeline" = the workflow's execution step
only (what every earlier round ran), "multiagent" = execution + the production per-table
verify/correct loop (`PKPEPerTableVerifyCorrectStep`, never run inside the cascade yet). This round:
Arm A `qwen3.8:27b`, Arm B / Tier 3 / judge `QWEN3.8-27B`, mode pipeline, all 9 gold papers
(dataset `pk_individual_qwen38_all`). F1_A and F1_B are scored from this run's own Arm A / Arm B tables (F1_B = the paper with Arm B's
table on every escalated table; n/a where nothing escalated). No stored collection numbers.
Job 8238170; work dir `.curate_agent_calibration/pk_individual_qwen38_pipeline/`; scratch
`cascade_v2r4_pki_pipeline-8238170_*`.

| pmid | sel | esc | F1_cascade | F1_A (this run) | F1_B (this run) | LLM calls | sec |
|---|---|---|---|---|---|---|---|
| 10971311 | 2 | – | 0.714 | 0.714 | n/a | 5 | 337 |
| 11849190 | 2 | – | 0.976 | 0.976 | n/a | 5 | 338 |
| 18426260 | 1 | – | 1.000 | 1.000 | n/a | 3 | 284 |
| 23200982 | 3 | – | 0.893 | 0.893 | n/a | 7 | 640 |
| 24989434 | 1 | – | 1.000 | 1.000 | n/a | 3 | 253 |
| 32056930 | 2 | – | 0.841 | 0.841 | n/a | 5 | 482 |
| 32153014 | 1 | – | 0.667 | 0.667 | n/a | 3 | 226 |
| 33253437 | 1 | – | 0.912 | 0.912 | n/a | 4 | 453 |
| 34746508 | 1 | 2 | 0.588 | 0.588 | 0.000 | 4 | 118 |
| **mean** | | | **0.844** | 0.844 | 0.000 (1 paper) | 4.3 | 348 |

Identical F1 to round 3 on all 9 papers: Arm A's output is the final output everywhere. Gate
results per table (Arm A): 32153014 t2 missed 13 of 15 → 0.155, 33253437 t2 missed 24 of 141 →
0.048, everything else ≤ 0.036; only 34746508 t2 (0.222) escalated, Arm B (pipeline mode)
returned 0 rows on the transposed case table as in rounds 2–3, settled by `unjustified_empty_loses`,
no judge call. Tier 3 this time reported 0 errors / 0 missed on 10971311 and 32056930 and 1
missed row on 23200982 t4 (round 3's gemma4 Tier 3 had said 34) - the Tier 3 verdict itself is
model- and run-dependent.

### Time cost of round 4

Total GPU time across the 9 tasks: 1 h 2 min (3776 s), one A100 each; a single
model served both arms, so no model swapping (round 3 with two models: mean 538 s per paper).

| pmid | tables | Slurm task wall-clock | cascade total (s) | select | Arm A | gate A | Arm B | gate B | judge |
|---|---|---|---|---|---|---|---|---|---|
| 10971311 | 2 | 00:06:54 | 336 | 122 | 203 | 11 | – | – | – |
| 11849190 | 2 | 00:06:53 | 338 | 118 | 209 | 11 | – | – | – |
| 18426260 | 1 | 00:05:59 | 284 | 118 | 157 | 9 | – | – | – |
| 23200982 | 3 | 00:11:53 | 639 | 120 | 494 | 26 | – | – | – |
| 24989434 | 1 | 00:05:26 | 252 | 119 | 125 | 7 | – | – | – |
| 32056930 | 2 | 00:09:15 | 481 | 119 | 347 | 15 | – | – | – |
| 32153014 | 1 | 00:04:59 | 225 | 119 | 91 | 15 | – | – | – |
| 33253437 | 1 | 00:08:50 | 451 | 130 | 282 | 40 | – | – | – |
| 34746508 | 1 | 00:02:47 | 117 | 12 | 50 | 9 | 29 | 18 | – |
| **mean** | | | **347** | 109 | 217 | 16 | 3 | 2 | – |

### Why Arm B (pipeline) returned 0 rows on 34746508 table 2 - root cause and fix (2026-10-31)

The table is transposed (parameter rows, Case 1-5 columns, blank header over the labels). The
workflow's parameter-type align step got the right answer from the model (`col_name="Unnamed: 0"`,
qwen3.8 and gemma4 alike) and refused it: commit 6e8669a (2026-10-06) added a name-based guard
that never transposes on a column called `Unnamed: N` / `ID` / `Subject` / `No.`, written for
33253437 where that column holds drug names. Untransposed, header categorization mapped Case 1-5
to Patient ID and nothing to Parameter value; SplitByColumnsStep dropped the labels column as
Uncategorized and returned `[]`; 0 rows, no error. The 2026-10-01 pipeline_qwen38 collection,
five days before the guard, had the table's 15 rows right.

Fix (option 2 of the three discussed): transpose when the model asks, then check the result.
`check_transpose_result` rejects a column whose values are not mostly distinct labels (33253437:
6 distinct over 47 -> 0.13; 34746508: 9 of 9 -> 1.0) or a result with < 3 columns, via
RetryException so the model is asked again with the reason; after the retries the table is kept
untransposed. Tests on both papers' real tables in
`tests/test_pk_ind_param_type_align_transpose_check.py`. This changes the production
pk_individual workflow (pipeline and multi-agent modes), not only the cascade.

**Live check of the fix** (job 8240699, 34746508 only, Arm B pipeline mode, work dir
`.curate_agent_calibration/pk_individual_qwen38_pipeline_fix2/`, scratch
`cascade_v2r4_pki_pipeline_fix2-8240699_8`): the align step now transposes table 2 into one row
per case; Arm B returns 15 rows (all 5 cases × mother's blood / cord blood / 6 months, null
findings included), Tier 3 on B: 0 errors, 0 missed ("All values match"); the LLM judge ran for
the first time on pk_individual and picked B ("Candidate B is the clear winner ... 0 errors and
0 missed rows, whereas Candidate A is marked 'incorrect' with 10 missed rows"). Paper F1
0.588 → **0.889** (F1_A 0.588, F1_B 0.889, both from this run); 5 LLM calls, 276 s. Gold has 12
rows, so the 3 extra "Not measured" maternal-blood rows for cases 1-4 cost the remaining precision.

## v2 per-table cascade — fifth live round, pk_individual, qwen3.8 both arms, with the transpose fix (2026-10-31)

Same configuration as round 4 (Arm A `qwen3.8:27b`, Arm B pipeline mode / Tier 3 / judge on
`QWEN3.8-27B`, all 9 gold papers) plus the parameter-type align fix above. Job 8240894; work dir
`.curate_agent_calibration/pk_individual_qwen38_pipeline_r5/`; scratch `cascade_v2r4_pki_pipeline_r5-8240894_*`.
Every F1 below is from this run (F1_B = the paper with Arm B's table on the escalated table).

| pmid | sel | esc | F1_cascade | F1_A (this run) | F1_B (this run) | round 4 | LLM calls | sec | task wall-clock |
|---|---|---|---|---|---|---|---|---|---|
| 10971311 | 2 | – | 0.714 | 0.714 | n/a | 0.714 | 5 | 340 | 00:06:49 |
| 11849190 | 2 | – | 0.976 | 0.976 | n/a | 0.976 | 5 | 355 | 00:07:06 |
| 18426260 | 1 | – | 1.000 | 1.000 | n/a | 1.000 | 3 | 304 | 00:06:15 |
| 23200982 | 3 | – | 0.893 | 0.893 | n/a | 0.893 | 7 | 574 | 00:10:49 |
| 24989434 | 1 | – | 1.000 | 1.000 | n/a | 1.000 | 3 | 234 | 00:05:11 |
| 32056930 | 2 | – | 0.841 | 0.841 | n/a | 0.841 | 5 | 490 | 00:09:24 |
| 32153014 | 1 | – | 0.667 | 0.667 | n/a | 0.667 | 3 | 203 | 00:04:40 |
| 33253437 | 1 | – | 0.912 | 0.912 | n/a | 0.912 | 4 | 424 | 00:08:22 |
| **34746508** | 1 | 2 | **0.889** | 0.588 | 0.889 | 0.588 | 5 | 264 | 00:05:46 |
| **mean** | | | **0.877** | 0.844 | 0.889 (1 paper) | 0.844 | 4.4 | 354 | |

Mean 0.844 → 0.877. The only change is 34746508: Arm B now curates table 2 (15 rows, Tier 3 0 errors
/ 0 missed), the LLM judge ran and picked B. The other 8 papers reproduced round 4 exactly - the fix
touched no other table, and Arm A was identical to round 4 on all 9 papers. Gate: still one
escalation in 9 papers; 32153014 t2 (13 missed of 15, rate 0.155) and 33253437 t2 (24 missed of
141, 0.048) stay under 0.2. GPU total 1 h 4 min, mean 354 s per paper; no errors in any task.

### Round 5 in the pk-individual benchmark comparison (2026-10-31)

Round 5's nine `combined.csv` files were staged as
`benchmark/data/pk-individual/2026-10-31-collect-cascade_qwen38/{pmid}_cascade-qwen38.csv` (model label
`cascade-qwen38`, new `LLModelType.CASCADE_QWEN38`), scored with the two benchmark tests (Slurm job
8241414: semantic `benchmark/test_pk_individual_benchmark_with_semantic.py`, F1
`benchmark/test_f1_with_semantic.py`), and integrated as the set `cascade_qwen38` - a fifth mode,
"Cascade" - into a new snapshot of the pk-individual benchmark comparison:
`/fs/scratch/PCON0100/feng1426/projects/playground/tabular-data-benchmark/pk-individual/benchmark_results/20261031-cascade-qwen38/`
(CSVs, README, `figures/` and `figures_open/` 01-09 regenerated; `integrate_cascade_qwen38.py` documents the
set and its time/token method). Against the 19 other complete sets: semantic mean 82.2 (mid-pack:
multiagent_gpt54 87.1, skills_qwen38 86.0, pipeline_qwen38 84.0, simple_qwen38 81.8), strict F1 macro
**0.877**, second only to skills_qwen38 (0.910) and above every pipeline / multi-agent set (best 0.855),
micro P/R 0.92/0.86; 284,683 tokens and 53 min for the 9 papers (pipeline_qwen38: 1.74 M tokens, 1 h 09 m;
simple_qwen38: 151 k, 37 min).

## pk_summary — sixth round, qwen3.8 for both arms (2026-10-31)

First pk_summary run with Arm A = `qwen3.8:27b` simple prompt and Arm B (pipeline mode) / Tier 3 /
judge = `QWEN3.8-27B` (rounds 1-2 used qwen3.6 for both). Dataset `pk_summary_qwen38` (the 9 gold papers
with source HTML; 30950674 has none), script `scripts/slurm/slurm_cascade_v2_pks_qwen38_array.sh`, job 8241591, work dir
`.curate_agent_calibration/pk_summary_qwen38_pipeline_r6/`, scratch `cascade_v2r6_pks_pipeline_r6-8241591_*`.
Every F1 is from this run (F1_B = the paper with Arm B's table on each escalated table).

| pmid | sel | esc | F1_cascade | F1_A (this run) | F1_B (this run) | round 2 (qwen3.6) | LLM calls | sec |
|---|---|---|---|---|---|---|---|---|
| 16143486 | 3 | 2 | 0.918 | 0.533 | 0.918 | 0.952 | 8 | 449 |
| 17635501 | 1 | – | 0.966 | 0.966 | n/a | 0.867 | 3 | 227 |
| 22050870 | 2 | 2 | 0.857 | 0.667 | 0.857 | 0.588 | 6 | 386 |
| 29943508 | 1 | – | 1.000 | 1.000 | n/a | 1.000 | 3 | 199 |
| 30825333 | 2 | 3 | 1.000 | 0.000 | 1.000 | 1.000 | 6 | 220 |
| 34114632 | 2 | 2 | 0.983 | 0.333 | 0.983 | 0.750 | 6 | 369 |
| 34183327 | 1 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | 5 | 403 |
| 35465728 | 1 | – | 1.000 | 1.000 | n/a | 1.000 | 3 | 245 |
| 35489632 | 1 | – | 1.000 | 1.000 | n/a | 1.000 | 3 | 190 |
| **mean** | | | **0.969** | 0.611 | 0.952 (5 papers) | 0.906 | 4.8 | 299 |

Mean 0.969 (round 2 with qwen3.6: 0.906), 5 of 9 papers escalated, Arm B won every escalation, and
the mechanism did exactly what it is for: Arm A alone would have scored 0.611. GPU total 54 min; 299 s
per paper. The escalations, from the reports:

| table | Arm A | gate on A | Arm B | gate on B | Tier 4 |
|---|---|---|---|---|---|
| 16143486 t2 | 0 rows (header-only reply) | assessment: has data, 16 expected | 16 rows | 0 err / 0 missed | rule `unjustified_empty_loses` → B |
| 22050870 t2 | 0 rows (header-only reply) | has data, 18 expected | 12 rows | 12 err / 0 missed | rule → B |
| 30825333 t3 | 0 rows (**parser lost 8 rows**) | has data, 8 expected | 8 rows | 0 / 0 | rule → B |
| 34114632 t2 | 0 rows (header-only reply) | has data, 12 expected | 23 rows | 0 / 1 | rule → B |
| 34183327 t2 | 30 rows, **all 30 dropped by the parser** | Tier 1 rate 0.5 | 30 rows | 14 err | LLM judge → B ("Candidate A's report indicates that all 30 rows were dropped as implausible ... Candidate B ... retained all 30 rows with a high average fit score of 8.73") |

(30825333 t2 came back empty too and the assessment agreed the table has no summary data - a justified
empty, kept with no escalation.)

### Why Arm A was so weak here, and what was fixed

Three causes, two of them ours:

1. **Parser: headerless rows one trailing empty too wide were lost silently.** 30825333 t3's reply was 8
   good rows with no header line and 17 fields each (a trailing comma). `_parse_headerless` requires two
   exact-width rows before it believes it has a CSV block, counted them before stripping trailing
   empties, found none, and returned nothing - with nothing in `dropped`, so Tier 1 saw a clean empty
   table. Fixed: trailing empties are stripped first, and when the block still fails the check every
   comma-bearing line is reported as dropped. Same change in the pk_individual parser.
2. **Parser: a header missing one name over full-width rows dropped every row.** 34183327 t2's reply
   had a 15-name header (no `Population`) over 17-field rows (one stray empty). `parse_csv` only
   switched to the canonical 16-column schema when the rows were exactly 16 wide, so all 30 rows went
   to `_realign` against the 15-name header and were dropped. Fixed: rows of width 16 or 17 count as
   full-width, the canonical schema is used, and `_realign`/`_reshift` place the stray empty (verified
   cell by cell on the real rows: Population, Interval type, limits and P value all land correctly).
3. **Model: qwen3.8 answered three tables with a header line and nothing else** (16143486 t2,
   22050870 t2, 34114632 t2) after 2,000-3,500 thinking tokens each; qwen3.6 curated all three in
   round 2. The runner only retried without thinking when the response was completely empty. Fixed:
   a reply that parses to no rows and no dropped rows is retried once with `think=False`, exactly as
   an empty reply is, and the model's thinking text is now kept next to the raw reply
   (`arm_a.thinking.txt`) so the next such case can be diagnosed from the file.

Offline regression over all 111 saved Arm A replies (every round, both schemas): only the two tables
in 1 and 2 change (30825333 t3: 0 → 8 rows; 34183327 t2: 0 rows / 30 dropped → 30 rows / 0 dropped).
Tests: `tests/test_simple_prompt_parsers.py` (the real rows of both tables, the dropped reporting, the
pk_individual analogue) and `tests/test_simple_prompt_curate_table.py` (header-only retry, thinking kept).
The think=False retry can only be checked live. Expected effect of a re-run: the same or better F1 at
fewer Arm B calls (three to five of the five escalations should no longer be needed).

### Round 6 in the pk-summary benchmark comparison (2026-10-31)

Round 6's nine `combined.csv` files were scored with the pk-summary comparison's own evaluator calls
(semantic `TablesEvaluator` + `TablesF1Evaluator`, GPU job 8241941) and added as the set `cascade_qwen38`
to a new snapshot
`/fs/scratch/PCON0100/feng1426/projects/playground/tabular-data-benchmark/pk-summary/benchmark_results/20261031-cascade-qwen38/`
(CSVs, README, `figures/` and `figures_open/` 01-09). Among the 19 sets: semantic 91.1 (joint first with
simple_gpt54; simple_qwen38 90.8, multiagent_qwen38 89.3, pipeline_qwen38 88.9), strict F1 macro
**0.969**, first (simple_gpt54 0.936, simple_qwen38 0.932, skills_qwen38 0.923), micro P/R 0.954/0.938;
304,894 tokens and 44 min for the 9 papers (pipeline_qwen38 372 k / 1 h 16 m; multiagent_qwen38 569 k /
45 min). Note that the stored `simple_qwen38` set (0.932) is not what Arm A produced here (0.611): that
collection's own notes say two of its papers were re-run by hand after a column shift and a zero-row
reply - the same two failure shapes the fixes above now handle in code.

## v2 per-table cascade — seventh round, pk_individual, worst-tier gate (2026-10-31)

Same as round 5 (Arm A `qwen3.8:27b`, Arm B pipeline mode / Tier 3 / judge `QWEN3.8-27B`, transpose fix)
plus the gate change: `error_rate` = max of the tier rates instead of their mean (see
docs/VERIFICATION_AGENT_DESIGN.md, "Gate rule: worst tier"). Job 8242232; work dir
`.curate_agent_calibration/pk_individual_qwen38_pipeline_r7/`; scratch `cascade_v2r4_pki_pipeline_r7-8242232_*`.
Every F1 is from this run.

| pmid | sel | esc | F1_cascade | F1_A (this run) | F1_B (this run) | round 5 | LLM calls | sec |
|---|---|---|---|---|---|---|---|---|
| 10971311 | 2 | – | 0.714 | 0.714 | n/a | 0.714 | 5 | 310 |
| 11849190 | 2 | – | 0.976 | 0.976 | n/a | 0.976 | 5 | 241 |
| 18426260 | 1 | – | 1.000 | 1.000 | n/a | 1.000 | 3 | 200 |
| 23200982 | 3 | – | 0.893 | 0.893 | n/a | 0.893 | 7 | 490 |
| 24989434 | 1 | – | 1.000 | 1.000 | n/a | 1.000 | 3 | 148 |
| 32056930 | 2 | – | 0.841 | 0.841 | n/a | 0.841 | 5 | 476 |
| **32153014** | 1 | 2 | **1.000** | 0.667 | 1.000 | 0.667 | 5 | 359 |
| 33253437 | 1 | – | 0.912 | 0.912 | n/a | 0.912 | 4 | 415 |
| 34746508 | 1 | 2 | 0.889 | 0.588 | 0.889 | 0.889 | 5 | 246 |
| **mean** | | | **0.914** | 0.844 | 0.944 (2 papers) | 0.877 | 4.7 | 320 |

The rule change did exactly what the offline recount predicted: one new escalation (32153014 t2, Arm A
15 clean rows / 13 missed → Tier 3 rate 0.464; Arm B 30 rows, 0 errors / 0 missed; judge picked B:
"Candidate B is the clear winner ... 0 errors and 0 missed rows, whereas Candidate A is marked as
incorrect with 13 missed rows"), no spurious ones, every other paper identical to round 5. Mean
0.877 → **0.914**; 2 escalations, both decided by the LLM judge, both right. GPU total 55 min, 320 s
per paper. Remaining gap to Arm B: 10971311 (Tier 3 reports 0 errors / 0 missed on both tables while
the stored pipeline_qwen38 set scores 0.990 there) - a Tier 3 recall problem the gate cannot fix.

### Round 7 in the pk-individual benchmark comparison (2026-10-31)

The `cascade_qwen38` set in `benchmark_results/20261031-cascade-qwen38/` (pk-individual) was replaced by
round 7's output (re-staged, re-scored with GPU job 8242412, re-integrated; README and all 18 figures
regenerated; `remove_cascade_qwen38_from_readme.py` + the updated `integrate_cascade_qwen38.py` /
`update_readme_cascade_qwen38.py` are the provenance). Among the 20 sets: semantic 84.4 (5th;
multiagent_gpt54 87.1, skills_qwen38 86.0), strict F1 macro **0.914, first** (skills_qwen38 0.910,
multiagent_gpt54 0.855, simple_qwen38 0.853), micro P/R 0.923/0.877; 361,085 tokens and 48 min for the
9 papers (pipeline_qwen38 1.74 M / 1 h 09 m).

## pk-pe pipelines - first live round of the eight other schemas (2026-11-02)

The cascade now covers all ten pipelines (docs/VERIFICATION_AGENT_DESIGN.md, "All ten pipelines").
First live run: `scripts/slurm/slurm_cascade_v2_pkpe_qwen38_array.sh`, job 8271896, one schema per array task,
Arm A = `qwen3.8:27b` on the new `prompts/simple_prompts/<schema>.md`, Arm B (pipeline mode) /
Tier 3 / judge = `QWEN3.8-27B`, worst-tier gate at 0.2. Papers = the pk-pe benchmark set (PMC XML
under `tabular-data-skills-route/.staging`), each schema on the papers that have that pipeline in
`benchmark/data/pk-pe/2026-4-10` or `2026-4-16`; pk_population_individual has none there and did
not run. Work dir `.curate_agent_calibration/pkpe_qwen38_pipeline/<schema>/`, scratch
`curate_agent_cascade/cascade_v2_pkpe_pipeline-8271896_<task>/`.

**There is no gold for these pipelines.** The pk-pe benchmark compares two model runs; the driver
scored each paper against the 2026-4-16 gpt-4o file and labels the column so. Those numbers are
not quality scores: where gpt-4o found nothing its file holds one all-"N/A" row while the cascade
writes an empty table (F1 0.000 for two "nothing" answers), the gpt-4o population-summary files
are the older 9-column shape, and free-text columns (Note, Sample time) follow different
conventions. They are kept only because the driver prints them.

| schema | pmid | units | esc | rows | F1 vs gpt-4o (not gold) | LLM calls | sec |
|---|---|---|---|---|---|---|---|
| pk_drug_summary | 18391836 | text | - | 0 (justified empty) | n/a | 3 | 149 |
| pk_drug_summary | 18665197 | text | - | 0 (justified empty) | 0.000 | 3 | 45 |
| pk_drug_summary | 18673410 | text | - | 22 | n/a | 2 | 252 |
| pk_drug_summary | 18686186 | text | - | 0 (justified empty) | 0.000 | 3 | 35 |
| pk_drug_summary | 18694901 | text | - | 4 | 0.000 | 2 | 129 |
| pk_drug_summary | 18701886 | text | - | 4 | 0.000 | 2 | 133 |
| pk_drug_summary | 18758393 | text | - | 0 (justified empty) | n/a | 3 | 61 |
| pk_drug_summary | 18762498 | text | - | 0 (justified empty) | n/a | 3 | 32 |
| pk_drug_individual | 18701886 | text | - | 11 | 0.000 | 3 | 324 |
| pk_specimen_summary | 18665197 | text | - | 3 | 0.000 | 2 | 243 |
| pk_specimen_summary | 18686186 | text | - | 4 | 0.000 | 2 | 96 |
| pk_specimen_summary | 18694901 | text | - | 1 | n/a | 2 | 152 |
| pk_specimen_summary | 18701886 | text | - | 2 | 0.500 | 2 | 118 |
| pk_specimen_summary | 18758393 | text | - | 2 | 0.400 | 2 | 96 |
| pk_specimen_summary | 18762498 | text | - | 1 | 1.000 | 2 | 63 |
| pk_specimen_individual | 18701886 | text | text | 12 | 0.000 | 4 | 348 |
| pk_population_summary | 18665197 | t1, t2 | - | 37 | 0.000 | 6 | 387 |
| pk_population_summary | 18686186 | text | - | 0 (justified empty) | 0.000 | 3 | 48 |
| pk_population_summary | 18694901 | t1 | - | 18 | 0.000 | 3 | 114 |
| pk_population_summary | 18762498 | t1 | - | 22 | 0.000 | 3 | 93 |
| pe_study_info | 18670224 | text | - | 1 | 0.000 | 2 | 200 |
| pe_study_outcome | 18670224 | t1, t2 | - | 74 | 0.906 | 5 | 367 |

("n/a" = the gpt-4o set has no file for that paper and pipeline; the paper comes from the qwen3.5
version's list.) GPU time per task (Ollama start-up included): 17:12, 8:26, 14:41, 7:36, 12:31,
5:06, 7:50 - 73 minutes for the 22 paper-schema pairs, 24 units.

Per unit, from the reports:

| schema | pmid | unit | Arm A rows (dropped) | Tier 2 unsupported / checked | Tier 3 on A | rate | escalated |
|---|---|---|---|---|---|---|---|
| pe_study_info | 18670224 | text | 1 (0) | 0/1 | 0 err / 0 missed | 0.00 | - |
| pe_study_outcome | 18670224 | t1 | 20 (0) | 0/67 | 0 / 0 | 0.00 | - |
| pe_study_outcome | 18670224 | t2 | 54 (0) | 0/130 | 0 / 0 | 0.00 | - |
| pk_drug_individual | 18701886 | text | 11 (0) | 0/11 | 0 / 2 missed | 0.15 | - |
| pk_drug_summary | 18391836 | text | 0 | - | justified empty | 0.00 | - |
| pk_drug_summary | 18665197 | text | 0 | - | justified empty | 0.00 | - |
| pk_drug_summary | 18673410 | text | 22 (1) | 0/55 | 1 err / 0 | 0.05 | - |
| pk_drug_summary | 18686186 | text | 0 | - | justified empty | 0.00 | - |
| pk_drug_summary | 18694901 | text | 4 (0) | 0/8 | 0 / 0 | 0.00 | - |
| pk_drug_summary | 18701886 | text | 4 (0) | 0/6 | 0 / 0 | 0.00 | - |
| pk_drug_summary | 18758393 | text | 0 | - | justified empty | 0.00 | - |
| pk_drug_summary | 18762498 | text | 0 | - | justified empty | 0.00 | - |
| pk_population_summary | 18665197 | t1 | 37 (0) | 0/74 | 0 / 1 missed | 0.03 | - |
| pk_population_summary | 18665197 | t2 | 0 | - | justified empty (definitions table) | 0.00 | - |
| pk_population_summary | 18686186 | text | 0 | - | justified empty (in-vitro study) | 0.00 | - |
| pk_population_summary | 18694901 | t1 | 18 (0) | 0/38 | 0 / 0 | 0.00 | - |
| pk_population_summary | 18762498 | t1 | 22 (0) | 0/58 | 0 / 0 | 0.00 | - |
| **pk_specimen_individual** | 18701886 | text | 12 (2) | 0/20 | 1 err / 8 missed | **0.45** | **yes** |
| pk_specimen_summary | 18665197 | text | 3 (0) | 0/12 | 0 / 0 | 0.00 | - |
| pk_specimen_summary | 18686186 | text | 4 (0) | 0/0 | 0 / 0 | 0.00 | - |
| pk_specimen_summary | 18694901 | text | 1 (0) | 0/4 | 0 / 0 | 0.00 | - |
| pk_specimen_summary | 18701886 | text | 2 (0) | 0/4 | 0 / 0 | 0.00 | - |
| pk_specimen_summary | 18758393 | text | 2 (0) | 0/1 | 0 / 0 | 0.00 | - |
| pk_specimen_summary | 18762498 | text | 1 (0) | 0/3 | 0 / 0 | 0.00 | - |

What the round showed:

- **Every stage ran on every schema**: XML sources, text and table units, the generic parser, the
  per-schema Tier 3 rules and the empty-table assessment. Seven of the 24 units were justified
  empties, each with a sensible reason from Tier 3 (e.g. 18665197 "measures exposure to dietary
  components ... no drug dosing regimen", 18686186 "in vitro study using rabbit liver microsomes",
  18758393 "a method validation paper"). Arm A's header-only answers were retried with
  `think=False` (the `arm_a: 2` call counts) and stayed empty, which the assessment then accepted.
- **One escalation, and it exposed an Arm B failure mode.** pk_specimen_individual on 18701886:
  Arm A wrote 12 rows (9 infants' meconium) and Tier 3 found 8 missing urine rows (rate 0.45).
  Arm B, the production `PKSpecIndWorkflow`, raised its "Wrong answer: []" RetryException and
  returned the one-row "N/A" placeholder the full-text workflows use for "nothing found";
  `normalize_to_schema` kept it as a 1-row table, Tier 3 on B said "All values match", and only the
  judge (reports-only pass) saved the pick: "Candidate B has only 1 total row ... effectively
  contains no valid data" -> A. Fixed after the run: `arms.drop_placeholder_rows` turns an
  all-placeholder Arm B table into an empty one, so the empty-table assessment runs on it
  (`tests/test_curate_arms.py::test_expensive_arm_placeholder_row_is_an_empty_table`).
- **Tier 3 recall on prose is the open question**, as on pk_individual: 24 units and one
  escalation is either a strong Arm A or a lenient verifier, and with no gold the round cannot
  tell. pe_study_outcome (the one table schema here) agrees with gpt-4o at F1 0.906 across 74
  rows, and pk_specimen_summary's three 18665197 rows match gpt-4o's counts (381 urine, 88 blood,
  361 vs 359 saliva) with the sampling window gpt-4o left empty filled in ("0-52 Week") - the
  only direct evidence of quality in this round.
- **Cost**: 24 to 300 s per Arm A call (the full text is 26-44k characters), 4 to 25 s per Tier 3
  call, 2 to 6 LLM calls per paper-schema pair.

Next: gold for a handful of pk-pe papers (or the pe gold under `benchmark/data/pe/baseline`, whose
papers have no local source yet) is what turns these reports into scores.
