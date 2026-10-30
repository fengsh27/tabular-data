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
