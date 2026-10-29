# Cascading curate agent: tiered validity checker + cross-model comparison

**Status: design, not yet implemented.** This doc specifies a new `CurateAgent`
orchestrator that sits between the existing `DesignAgent` (identification +
design) and output, replacing a single fixed-pipeline run with a cost-aware
cascade: cheap pass first, escalate only on detected failure, pick the better
of two independently-produced results when escalation happens.

## Context

Today, `PKPEManager.run()` (`extractor/agents_manager/pk_pe_manager.py:368`)
always does identification + design, then dispatches straight to one
fixed-cost pipeline run (multi-agent mode by default, or "pipeline mode" via
`--no-verify-correct`). There's no cheap-first-pass option in that path, and
no mechanism anywhere that runs two independent attempts and picks the better
one — confirmed by grep across `extractor/` for `compare`/`better`/`ensemble`/
`select_best`: zero hits related to result comparison.

Separately, `scripts/simple_prompt/run_simple_prompt.py` already implements a
cheap, single-call-per-table mode (no decomposition, no verification), but it
has **no retry or escalation logic at all** — a malformed row is dropped,
logged to `manifest.json`, and the run moves on
(`run_simple_prompt.py:216-285`, confirmed no `while`/retry loop around
`parse_csv`).

The goal: wire these two existing, previously-separate code paths together
with a validity gate in between, instead of building a third, parallel
execution engine.

```
DesignAgent (existing: PKPEIdentificationStep + PKPEDesignStep)
      │
      ▼
CurateAgent (new orchestrator)
  1. run_simple_prompt.py  (qwen3.8, single call)
  2. ValidityChecker  ──────────────► pass → done, return candidate A
  3. too many errors? ──► escalate
  4. PKPEManager multi-agent mode  (gemma4, has its own internal
     execution→verify→correct loop already)
  5. ValidityChecker on candidate B too (same checks, same report shape)
  6. ComparisonJudge  ──► pick winner, attach both reports
      │
      ▼
   Output: winning table + validity reports + judge rationale
```

## Why these two models/modes specifically

Per `.env.template` and `extractor/agents/agent_factory.py:29,68`,
`PIPELINE_LLM`/`AGENT_LLM` recognize `GPT-OSS-20B, QWEN3-30B, QWEN3.5-27B,
QWEN3.6-35B, QWEN3.8-27B, OPENAI, OPENAI-5, GEMMA4-31B`. Invocation for each
arm of the cascade (confirmed against existing benchmark result dirs and
`app_script_pmids.py`):

```bash
# Arm A — cheap pass
poetry run python scripts/simple_prompt/run_simple_prompt.py \
  --prompt prompts/simple_prompts/pk_summary.md --schema pk_summary \
  --backend ollama --model qwen3.8:27b --temperature 0

# Arm B — escalation (multi-agent mode is the *default* PKPEManager
# behavior — no --no-verify-correct flag — so it already runs its own
# internal execution → verify → correct loop, up to MAX_PER_TABLE_STEP_COUNT=5
# rounds per table, extractor/constants.py:132)
PIPELINE_LLM=GEMMA4-31B AGENT_LLM=GEMMA4-31B \
  poetry run python app_script_pmids.py -f ./data/pmids.csv -o ./out -p pk_summary
```

**Important cost note for the design:** Arm B is not "one extra gemma4 call."
It's a full multi-agent run with its own internal self-verification loop. The
escalation decision should be made knowing it's binary (cheap vs. expensive
tier), not assuming linear cost.

## ValidityChecker — three tiers, cheapest first

Each tier only runs on what the previous tier didn't already rule invalid on
its own (e.g. a zero-row table never reaches tier 2). All tiers run against
**both** candidates with the same code path, so their reports are directly
comparable by the `ComparisonJudge`.

### Tier 1 — Structural (pure code, already half-built)

`_fits()`/`_realign()` in `scripts/simple_prompt/_common.py:157-224` and
`_common_summary.py:88-157` already score per-cell type/vocabulary
plausibility (numeric-shape regex, closed-vocabulary membership for
`Population`/`Summary statistics`/etc., text-vs-numeric column checks) — this
is a genuine plausibility scorer, not just a field-count check. Today it's
used **only internally**, to pick which gap-insertion candidate is least bad
during parsing, and it refuses to guess on ties (row dropped,
`_MAX_GAP = 3` bounds how much it will even attempt, lines 189/108).

**Gap to close:** this scorer is private to the parse step and never exposed
as a standalone "score this already-produced table" function, and it only
exists in the `scripts/simple_prompt/` world — the multi-agent arm's output
never goes through `parse_csv()` at all, so there's currently no equivalent
check for candidate B. Phase 0 below extracts a shared, importable
`score_table_validity(csv_path, schema)` usable post-hoc on **either**
candidate's final CSV, reusing the existing `_fits` vocabulary/regex tables
rather than re-deriving them.

Signal produced: `dropped_rows` (already written to `manifest.json` for Arm
A; needs computing fresh for Arm B), `total_rows`, and a per-row fit score.

### Tier 2 — Provenance (pure code, already built — just needs wiring)

`skills/pk-pe-curation/curation-common/scripts/verify_provenance.py` is the
canonical copy (confirmed byte-identical to the 10 duplicated copies under
`ollama_skills/`, which is a generated flat mirror, not a second source of
truth — `git log` shows both trees touched by the same commit, `d00f852`).

It has **two modes**, and the distinction matters a lot for this design:

- `check_csv()` (line 241, the default): does every number in the curated CSV
  appear *anywhere* in the source documents? Pure existence check — catches
  hallucinated values, **not** column-shift/misattribution, since a shifted
  value is still a real number from the paper, just in the wrong cell.
- `check_attribution()` (line 168, `--attribution` flag): tokenizes each
  row's label columns (e.g. `Analyte`/`Specimen`/`Population`/`Parameter
  type`), finds the source table's row/column group with the best Jaccard
  token overlap (`build_groups`, line 138), and flags a number as
  **misattributed** if it exists in the source overall but not within that
  best-matching group's numbers.

`check_attribution` is the mechanism that would have caught the diagnosed
`30825333` column-shift bug (documented in
`docs/SIMPLE_MODE_QWEN38_ISSUES.md`) — **use attribution mode, not the
default existence mode**, as the Tier 2 check. Both functions already return
a structured `findings` list (`{row, column, cell, number[, row_labels]}`),
not just a bool, which plugs directly into the error-count aggregation below.

**Gap to close:** `verify_provenance.py` is currently a CLI script
(`main(argv)`); `check_csv`/`check_attribution` are already plain importable
functions, so the only work here is a thin adapter module instead of
shelling out, plus confirming/declaring `--label-columns` per pipeline (known
good for `pk_summary`: `Analyte, Specimen, Population, Parameter type`; needs
the same check for `pk_individual` before this tier covers that pipeline).

### Tier 3 — Plausibility (new, small, deferred to phase 2)

Cross-field consistency that's cheap to encode as a lookup table but isn't
covered by tiers 1–2: unit-vs-parameter-type compatibility (a half-life
shouldn't carry a concentration unit), gross magnitude sanity. Scoped as a
phase-2 addition — tiers 1–2 already cover both concretely diagnosed bug
classes (zero-rows/malformed rows, and column-shift misattribution), so this
tier isn't blocking.

### Tier 4 — LLM semantic check (expensive, last resort only)

Reuses the `PKPECommonStep`/`get_common_agent()` pattern and a prompt
structurally similar to `PKPECuratedTablesVerificationStep`
(`extractor/agents/pk_pe_agents/pk_pe_verification_step.py`), but judging
**two** candidate tables against the source rather than one table in
isolation. Only invoked when tiers 1–2 leave the gate or the comparison
genuinely undecided (both candidates clean, or both candidates equally
flawed) — never as the primary mechanism, so the usual failure mode of
LLM-judge unreliability stays confined to the narrow case it's unavoidable
for.

## Error score and the escalation gate

```
error_rate = (tier1.dropped_rows + tier2.unsupported_findings) / tier1.total_rows
escalate = error_rate > THRESHOLD  or  tier1.total_rows == 0 (on a paper with a real table)
```

`THRESHOLD` should **not** be guessed — pk_summary already has collected
`manifest.json` + per-paper outcomes for `simple_qwen38` and
`multiagent_gemma4` in `benchmark_results/`. Phase 5 calibrates the threshold
against that existing data (does it escalate on the two known-bad papers and
stay quiet on the known-good ones?) before trusting it in a live cascade.

## ComparisonJudge

1. If one candidate's `error_rate` is strictly lower → it wins, no LLM call.
2. If tied (including both-zero) → Tier 4 LLM comparison, given both
   candidates' validity reports (not the raw tables first — only pull full
   tables into the prompt if the reports alone can't disambiguate).
3. Output: winning table, both `ValidityReport`s, and (if Tier 4 fired) the
   judge's explanation — all written to the output manifest for auditability,
   mirroring how `PKPECuratedTablesVerificationStep` already writes
   `explanation`/`suggested_fix`.

## Where the new code lives

This is an orchestrator gluing two previously-unrelated code paths (a
standalone CLI script and a LangGraph-based manager) — it does **not** need
to become a third LangGraph graph. Keep it a plain, importable Python
orchestrator:

```
extractor/curate_agent/
  validity_checker.py      # Tiers 1–3, wraps _fits()-derived scorer + verify_provenance
  comparison_judge.py       # tie-break rule + Tier 4 LLM call
  curate_orchestrator.py    # the cascade itself: run A → check → gate → run B → check → judge
scripts/simple_prompt/
  validity.py                # NEW: extracted, standalone score_table_validity() (phase 0)
skills/pk-pe-curation/curation-common/scripts/
  verify_provenance.py       # existing, reused as-is via import, no changes needed
```

`ValidityReport` contract (JSON-serializable, written next to the existing
`manifest.json`):

```json
{
  "tier1": {"dropped_rows": 0, "total_rows": 14, "avg_fit_score": 2.1},
  "tier2": {"checked_numbers": 42, "unsupported": 2, "findings": ["..."]},
  "error_rate": 0.14,
  "escalate": true
}
```

## Implementation plan

**Phase 0 — extract reusable primitives (no behavior change yet)**
- Pull the `_fits()`-based plausibility scorer out of `_common.py`/
  `_common_summary.py`'s parse-time-only path into a standalone
  `score_table_validity(csv_path, schema)` in a new `scripts/simple_prompt/
  validity.py`, parametrized by the existing `COLS`/`_VOCAB` tables per
  schema. Parsing itself keeps using `_fits`/`_realign` unchanged; this just
  exposes the same scoring as a post-hoc, importable function.
- Write a thin import adapter for `verify_provenance.check_attribution`/
  `check_csv` (no changes to that file).

**Phase 1 — ValidityChecker, tiers 1–2 only**
- New `extractor/curate_agent/validity_checker.py` combining both signals
  into the `ValidityReport` shape above.
- Regression test: run it against the two already-diagnosed `simple_qwen38`
  failures (`30825333` column-shift, `34183327` zero-rows,
  `docs/SIMPLE_MODE_QWEN38_ISSUES.md`) and confirm both are flagged — these
  are real, already-understood failures, so they make a precise acceptance
  test for "does the checker actually catch what it's meant to."

**Phase 2 — CurateAgent orchestrator, cascade without comparison**
- `extractor/curate_agent/curate_orchestrator.py`: run Arm A, check, gate; on
  escalate, run Arm B via `PKPEManager`, check. Interim pick-rule (lower
  `error_rate` wins) — no Tier 4 yet.

**Phase 3 — ComparisonJudge, Tier 4 tie-break**
- New prompt + step modeled on `PKPECuratedTablesVerificationStep`, reusing
  `PKPECommonStep`. Wire into the orchestrator's tie case only.

**Phase 4 — Tier 3 plausibility rules (optional)**

**Phase 5 — Calibration and benchmark validation**
- Use existing `benchmark/configs.py` + `TablesEvaluator`/`TablesF1Evaluator`
  scoring against the already-collected `simple_qwen38` and
  `multiagent_gemma4` pk_summary results to pick `THRESHOLD`, then run the
  full cascade end-to-end and confirm it beats either arm alone on a
  cost-adjusted basis (not just raw F1 — factor in how often it actually
  escalates).

## Open questions

- **Scope for v1**: `pk_summary` only (most mature/benchmarked pipeline,
  `Analyte`/`Specimen`/`Population`/`Parameter type` label-columns already
  known-good for attribution mode), or all pipelines from the start? Per-
  pipeline label-column config for Tier 2 needs confirming for each one
  before it's covered.
- **Does Arm B's own internal verify/correct loop make our external Tier 1–2
  check redundant for candidate B?** No — that loop is an LLM self-check
  against the paper's own source tables (`PKPECuratedTablesVerificationStep`,
  `pk_pe_verification_step.py`), not a deterministic/provenance check. Worth
  stating explicitly so it isn't "optimized away" later under the assumption
  multi-agent output is already validated.
- **Threshold granularity**: one global `THRESHOLD`, or per-pipeline
  (pk_summary vs. pk_individual likely have different baseline noise
  floors)?

## Related files

- `docs/SIMPLE_MODE_QWEN38_ISSUES.md` — the two diagnosed `simple_qwen38`
  bugs this design is built to catch.
- `docs/TODO_QWEN38_ISSUES.md`, `docs/ISSUES_SUMMARY_20261008.md` — prior
  cross-pipeline hardening context.
- `benchmark_results/20261007-baseline-fix/` (scratch) — existing
  `simple_qwen38`/`multiagent_gemma4` pk_summary results to calibrate against
  in Phase 5.
