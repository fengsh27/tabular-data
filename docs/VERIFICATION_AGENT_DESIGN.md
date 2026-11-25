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

## v2 — per-table cascade (implemented)

The v1 orchestrator above was **paper-level**: Arm A on every table, one Tier
1–3 check on the whole `combined.csv` against all source tables at once, one
bad table re-curating the entire paper with multi-agent mode, and the judge
only on an exact `error_rate` tie. Live calibration (`docs/cascade_mode_results.md`)
showed the cost of that shape on PMID 16143486: the whole-paper Tier 3 prompt
took 3m51s and mis-reported Table 4's 32 individual measurements as "missing"
from a pk_summary table, and the lower `error_rate` picked the worse arm
(F1 0.76 vs 0.98) while the judge never ran.

`extractor/curate_agent/curate_orchestrator.py` now runs:

```
route    PKPEIdentificationStep -> PKPEDesignStep        2 LLM calls (skipped when schemas= is given)
select   select_pk_summary_tables()                      1 call per schema (same selector multi-agent uses)
per table
  Arm A  arms.run_cheap_arm_table   (run_simple_prompt.curate_table, in-process)   1 call
  Tier 1+2+3 against THIS table's source only
  escalate? -> Arm B  arms.run_expensive_arm_table   PKSumWorkflow/PKIndWorkflow.go_md_table
                      execution only, no internal verify/correct loop          10-14 calls
               Tier 1+2+3 on Arm B (Tier 1 rate excluded)
               ComparisonJudge (Tier 4) - ALWAYS when Arm B ran               1-2 calls
combine  per-table winners in source order -> <work_dir>/<schema>/combined.csv (+ cascade_report.json)
```

Decisions and their evidence:

- **Arm B is execution-only.** Multi-agent's internal verify/correct loop,
  measured as `multiagent` vs `pipeline` collections (same workflow, with vs
  without the loop), moves strict macro F1 by +0.005 on pk-summary and
  −0.002 on pk-individual, at 1–25 extra LLM calls per table. The cascade's
  own Tier 3 + judge take over that role.
- **Judge runs whenever Arm B ran**, not only on a tie (16143486 above).
- **Tier 3's completeness rules are schema-aware**
  (`llm_verification.completeness_rules_for`): per-subject rules for
  `pk_individual` (PMID 34746508 needed them), summary-level rules for
  `pk_summary` (16143486's gold has only Table 4's 4 mean rows).
- **Arm B's pk_summary columns are normalized to the canonical schema**
  (`arms.PK_SUMMARY_MULTIAGENT_TO_CANONICAL`, the same pairs
  `benchmark/pk_preprocess.py` applies), so Tier 1's by-name lookups see real
  values and winners from either arm concatenate.
- **Scope**: `arms.SUPPORTED_SCHEMAS` = pk_summary, pk_individual (the two
  with a simple prompt). Other pipelines the route step selects are returned
  under `not_handled`.

Nominal LLM calls per paper (T tables, S selected, m failing):
not escalated `3 + 2S` (v1: `T + 1`); escalated `3 + 2S + m × [(10–14) + 1 + (1–2)]`
(v1: `T + 2 + [1 + S×(10–14) + loop 1…25]`). 16143486 (T=S=4): v1 ≈ 48–75,
v2 ≈ 25–30 with one failing table.

Tools: `scripts/calibrate_validity_threshold.py --live --per-table` prints
the per-table gate values (Arm A re-run per selected table) so `THRESHOLD`
can be re-picked for per-table denominators; `scripts/run_cascade_benchmark.py`
runs the cascade live on a dataset and scores `combined.csv` against gold.

### Fixes from the first live round's Tier 4 audit (2026-10-31)

Scoring the six live judge verdicts against gold (swap the escalated table's
winner, re-score the paper) found 3 of 6 wrong, each for a reason one level
below the judge. Three fixes, in dependency order:

1. **Count-correct column shifts are repaired at parse time**
   (`scripts/simple_prompt/_common_summary.py` / `_common.py`, `_reshift`).
   `_realign` only ran on a field-count mismatch. On PMID 35465728 table 2 the
   model dropped the empty `Pregnancy stage` cell and added an empty after
   `median`, so 16 fields matched the 16 columns while `60 / Initial Cavg /
   12.9 / mg/L / median` sat one column left of home; F1 0.000, and Tier 3
   reported no error. `_reshift` tries every "move one empty cell" edit (up to
   two moves), scores candidates with `_fits_strict` - `_fits` with two
   changes that only matter inside the search: a cell that starts with a
   digit but goes on in words ("95% CI", "3 Month to < 3 Years") is text, and
   a label outside a column's vocabulary ("parturients" for Population) votes
   0 rather than -1, so the search cannot "earn" a move by parking a label in
   a free-text column - plus a row rule (`Lower limit` and `High limit` are
   filled together or not at all). It accepts only a candidate that beats the
   original, leaves no hard misfit, and wins outright on (score, fewer moves,
   fewer cells displaced). Anything else leaves the row as written for Tier 1
   to flag. In the same pass, `_realign`'s tie rule (two insert positions
   with equal fit → drop the row) gained a tie-break toward extending an
   existing run of empties - the slip it repairs - after a fresh reply lost 4
   of 8 rows to a tie between "Vancomycin,,Vancomycin" and "Vancomycin,
   Vancomycin,,". Re-parsing all 85 saved Arm A replies changed 7 tables;
   every change was checked row by row against gold (see
   cascade_mode_results.md).
   A deterministic Tier 4 rule sits in front of the judge
   (`curate_orchestrator._rule_verdict`): an empty candidate whose assessment
   says the source has curatable data loses to a candidate with rows, no LLM
   call; a justified empty Arm B against an Arm A with rows skips the
   reports-only pass and goes straight to the full-table judge
   (`compare(..., force_full_tables=True)`).
2. **An empty table can be the correct answer**
   (`llm_verification.assess_empty_table`, `validity_checker.check_validity`,
   `comparison_judge`). Every zero-row table used to score `error_rate 1.0`
   with no LLM call, and the judge read that as "complete failure". On
   18426260 table 2 (a "Mean ± SD (Range)" table, no patient identifiers) and
   34746508 table 3 (cases from other published studies) Arm A's empty table
   was right and Arm B's filled table was all false positives. The gate now
   spends one call on an empty table asking whether the source table holds
   data of this schema's kind at all, with per-schema applicability rules
   (group statistics and literature-review tables are not pk_individual data;
   the latter is annotation policy and is stated in the prompt). "No" →
   `error_rate 0`, not escalated, `tier3.empty_table_justified`; "yes" →
   failure as before, with the rows that should have been curated recorded.
   The judge prompts no longer claim the rates are "tied" (false since v2),
   state that zero rows is correct when the source has no curatable data, and
   `_format_report` prints per-tier findings plus the empty-table assessment
   but not the composite `error_rate` - in all six live verdicts the judge did
   nothing but compare that number, whose two sides are not even on one scale
   (A's includes Tier 1, B's excludes it).
3. **pk_individual gets its own table selector**
   (`table_utils.select_pk_individual_tables`, dispatched by
   `curate_orchestrator.select_tables`). Both schemas used the summary
   selector ("any PK data"), which is why the two tables above were curated
   at all. The new prompt asks for per-individual values measured in this
   study and excludes group-statistics tables and tables of other papers'
   cases. The production multi-agent `PKIndividualTablesCurationTool` still
   uses the summary selector; switching it moves that benchmark's baselines
   and is a separate decision. `run_cascade_benchmark.py --select-only` checks
   a selector on a dataset for one call per paper.

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


### Gate rule: worst tier, not the mean (2026-10-31)

`check_validity`'s `error_rate` is now the maximum of the tiers' failure fractions instead of their
mean. Reason, from pk_individual rounds 4-5: on PMID 32153014 table 2 Arm A returned 15 correct rows
and omitted 13 (the placenta and placenta-to-maternal columns); Tier 3 reported all 13 missing rows
(rate 13/28 = 0.46) but Tier 1 and Tier 2, which can only judge rows that exist, were 0.00, so the
mean was 0.155 and the table was never escalated. Each tier catches a different failure (malformed
rows, numbers not in the source, wrong or missing rows), so a failure on any one is a reason to run
Arm B. Recomputed over every table of the latest pk_individual and pk_summary rounds, the maximum
adds exactly two escalations (32153014 t2 and pk_summary 22050870 t3, both Tier 3 flags of 0.38-0.46)
and no spurious ones; no other table has any tier above 0.15. The 0.2 threshold is unchanged.

## All ten pipelines: per-unit cascade and the generic simple prompts (2026-11-02)

Until this change the cascade ran only `pk_summary` and `pk_individual`, the two pipelines that
had a simple prompt and a hand-written parser. The other eight now run through the same code
path; what changed is below. Nothing in the two existing paths moved: `_common.py`,
`_common_summary.py`, their prompts, Tier 1 rules and thresholds are untouched, and the
pk_individual / pk_summary tests pass unchanged.

### Unit of work: table or text

The production tools do not all curate tables. `SKILLS_INTRO.md`'s split holds for the LangGraph
workflows too (`extractor/agents/pk_pe_agents/pk_pe_agent_tools.py`):

| pipelines | production input | cascade unit (`schema_specs.UNIT_KINDS`) |
|---|---|---|
| pk_summary, pk_individual, pe_study_outcome | one selected table per `go_md_table` call | `table` - one unit per selected table |
| pk_drug_summary/individual, pk_specimen_summary/individual, pe_study_info | the article text, one `go_full_text` call (`FullTextCurationTool`) | `text` - one unit per paper |
| pk_population_summary/individual | each selected demographic table as "markdown + caption" through `go_full_text`, or the article text when the selector finds none | `table_or_text` |

`curate_orchestrator.build_units()` produces the units; `_curate_table()` (name kept) runs the
unchanged Arm A -> Tier 1/2/3 -> Arm B -> Tier 3 -> Tier 4 sequence on each. A text unit's source
file is `<work_dir>/source_text/full_text.md`, built by `article_text_of()` exactly as the
production tools build theirs (sections joined, HTML and tables stripped, references cut).
Records keep the per-table keys; a text unit has `table_index` 0 and `unit_kind` "text".
Selectors: `select_pe_tables` for pe_study_outcome, `select_pk_demographic_tables` for the
population pipelines (both are what the production tools call).

### Schemas as data

`scripts/simple_prompt/schema_specs.py` describes the eight schemas (columns in the production
workflows' post-rename order, header aliases, numeric / text / required columns, vocabulary
tie-breakers, unit kind). `_common_generic.SchemaParser` is `_common_summary.py`'s parser with
those roles as parameters (realign, reshift, headerless block, dropped reporting - the same
algorithms). `schema_registry.load_schema_module()` returns the hand-written module for the two
legacy schemas and a `SchemaParser` otherwise; `run_simple_prompt.py`, `validity.py` and
`arms.schema_common()` all go through it. Tier 1 scores the generic schemas with the pk_summary
rule (numeric cells must start with a number, required identifiers must be present and not a bare
number, no vocabulary reward; `Patient ID` is scored like pk_individual's: short is plausible),
with the low-fit threshold at -1 like pk_summary.

Two deliberate departures from the benchmark configs: `pk_population_summary` uses the
workflow's 15 columns (benchmark/configs.py still lists a 9-column set with a Patient ID), and
`pe_study_outcome` spells its third column `Outcome` as the ver2 workflow does (the benchmark's
preprocess maps it onto the gold's `Outcomes`).

### Prompts

`prompts/simple_prompts/<schema>.md` for the eight schemas, each: header + `---` + body with the
`{TABLE}` slot (the markdown table for a table unit, the article text for a text unit). Column
rules are taken from the production agents' prompts (`*_info_agent.py`, `*_refine_agent.py`):
the population / pregnancy-stage categories, "age only when explicitly stated", dose amount
with the unit split off, sample times numeric with a separate unit, count + percentage as value
+ variation for the population and PE outcome tables, one row for pe_study_info. All write an
empty cell (never N/A) for what the source does not state, so the parsed tables match the two
legacy schemas and the benchmark scorers.

### Tier 2, Tier 3 and the judge per schema

- Tier 2 (existence) on a generic schema checks only the schema's numeric columns (plus
  `Sample time`): label categories ("Trimester 3", "95% CI") and verbatim `Note` excerpts carry
  digits that are not source values. pk_summary / pk_individual keep the all-columns check they
  were calibrated with.
- Tier 3 gets a per-schema domain string (`arms.domain_for`), per-schema completeness rules
  (`llm_verification.TEXT_COMPLETENESS_RULES`: one row per dosing regimen / sampling scheme /
  characteristic statistic / outcome number; exactly one row for pe_study_info) and per-schema
  empty-table applicability rules (`TEXT_APPLICABILITY_RULES`). The empty-table prompt and the
  judge prompts say "source table or text".
- Arm B: `arms.get_expensive_workflow` builds the schema's production workflow;
  `run_expensive_arm_table` calls `go_md_table` or `go_full_text` per `arms.WORKFLOW_INPUT`,
  passing a table unit of a full-text workflow as "markdown + caption" like the population tool.
  `normalize_to_schema` maps pre-rename workflow spellings to the canonical columns.

### Evaluation data

There is no manual gold for the eight pipelines. The pk-pe benchmark
(`benchmark/test_pk_pe_benchmark_with_semantic.py`) compares two model runs
(`benchmark/data/pk-pe/2026-4-10` qwen3.5 vs `2026-4-16` gpt-4o). `scripts/calibrate_validity_threshold.py`
therefore defines `pkpe_<schema>_qwen38` datasets (papers = those with that pipeline in either
version; sources = the PMC XML under `tabular-data-skills-route/.staging`, which
`HtmlTableExtractor` parses) with a `reference` (the gpt-4o files) instead of a `gold_dir`.
`run_cascade_benchmark.py` scores against the reference when there is no gold and labels every
F1 column with what it was scored against. The live run is `scripts/slurm/slurm_cascade_v2_pkpe_qwen38_array.sh`
(one schema per array task). Results: docs/cascade_mode_results.md, "pk-pe pipelines".
