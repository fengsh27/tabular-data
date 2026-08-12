# TODO — `pk-pe-route` stability

Status of the routing-instability investigation on `wip/fengsh/fix-route-issues`.
Written 2026-08-11.

## The problem

`pk-pe-route` gave **different answers on repeated runs of the same paper**. Same
prepared assets, same prompts, same model — run it five times, get five different
pipeline selections. Routing is the gate for every curation pipeline, so an
unstable route makes every downstream result unreproducible, and makes any prompt
edit unmeasurable: you cannot tell an improvement from a reroll.

Measured over 110 papers × 5 repeats (550 route calls): only **24%** of papers
produced the same pipeline set across all five runs.

## Root cause: sampling temperature

The model ran at its default temperature (random token sampling). Switching to
**greedy decoding (temperature 0)** removes the run-to-run randomness.

Implemented as a derived Ollama model, because the Claude Code CLI has no
`--temperature` flag — the setting travels attached to the model name:

```
/users/PCON0100/feng1426/ollama/qwen36-t0.Modelfile   ->  qwen3.6:35b-t0
    FROM qwen3.6:35b
    PARAMETER temperature 0 / top_p 1 / top_k 1 / seed 42
```

A Modelfile `PARAMETER` is only a *default* — a client that sends its own
temperature silently overrides it, and that failure is invisible in the results.
A preflight probe (6 identical prompts per model) confirmed it is honored:
`qwen3.6:35b-t0` → 1/6 distinct replies, `qwen3.6:35b` → 2/6.

## Results

189 papers × 5 repeats. **Group A** is the controlled before/after — same papers,
same assets, same frozen pre-fix prompts, only sampling changed. **Group B** is
held-out papers the finding did not come from, so it has no before-arm.

| metric | Group A before | Group A after | Group B after |
|---|---|---|---|
| identify unanimous | 59/100 (59%) | **94/100 (94%)** | 84/89 (94%) |
| design unanimous | 24/100 (24%) | **75/100 (75%)** | 69/89 (78%) |
| **selected unanimous** | **24/100 (24%)** | **74/100 (74%)** | 71/89 (80%) |
| selected mean pairwise Jaccard | 0.731 | **0.946** | 0.965 |

Every one of the 10 Group A manifests improved; none regressed. The worst
(`manifest_3`, 3/10 identify) improved the most (10/10) — the signature of noise
being removed rather than a few groups carrying an average. Group B matches
Group A, so the effect is not specific to the original sample.

**Stability is not correctness.** Greedy decoding makes a wrong label repeat
perfectly. These papers have no reference labels, so none of this measures
accuracy. What it buys is a **zero noise floor**: a later prompt edit now
produces a signal attributable to the edit.

Run artifacts (not in the repo):
`/fs/scratch/PCON0100/feng1426/projects/playground/tabular-data-skills-route/`

## Done

- [x] **Bullet-3 contradiction in `01_identify.md`** — commit `fb353c0`. Narrowed
      the compartment-comparison rule and replaced the escape hatch with a
      purpose test. **Untested** — measured runs used the frozen pre-fix snapshot
      so temperature stayed the only variable.
- [x] **`design.json` schema unpinned** — `02_design.md` now names the exact three
      keys. The prompt previously showed `pipeline_tools` only by example, while
      the dispatch section right below says `selected_pipelines.json` three
      times; ~3% of calls were primed into naming the key `selected_pipelines`.
      **Untested** — needs the next route run to confirm it drives them to zero.
- [x] **Stability driver could not read those files** — `run_route_stability.py`
      now accepts either key and either shape, and records a `design_schema`
      column so drift shows up as data. Verified: reads 961/961 design files, up
      from 934.

### The `design.json` issue was a measurement bug, not a production bug

Worth recording so it is not re-opened. Nothing reads `design.json`
programmatically: labels reach `pipeline_skill_map.py` as command-line arguments,
and the orchestrator reads only `selected_pipelines.json` (all 968 of which used
one identical schema). Checked across 956 calls, `design.json` and the dispatch
agreed **956/956, zero disagreements** — the file always described what actually
happened, whichever key it used. No paper was ever misrouted by this.

What it did cost was three wrong numbers, since a missing key read as a missing
artifact and then as instability:

| reported | actual |
|---|---|
| design unanimous 72% | **75%** |
| "9% mechanical artifact loss" | **~2%** |
| driver artifact fix = substantial work item | a two-line read |

`design.json` earns its keep as the audit trail that lets that 956/956
cross-check be run at all. Keep it readable.

## Curation batches were losing output they had already produced

Separate from routing; found while checking the runtime estimate. The `qwen3.6`
batch over 203 papers reported **82% of selected pipelines succeeded**. That was
a collection failure, not a curation failure.

Skills end with `OUT="${SKILL_OUTPUT_FOLDER:-.}"`. When the model writes the CSV
with its editor tool rather than running that snippet, the variable never
expands, the `:-.` fallback wins, and the file lands under the CWD — the scratch
dir. `run_curation()` saw nothing at the expected path and recorded a failure
while the rows sat one directory away.

| | pipelines | rows |
|---|---|---|
| reported succeeded | 671 / 816 (82%) | |
| recovered from scratch (`e3f9af6`, applied) | +78 → 749 (92%) | 1,995 |
| still recoverable from each skill's stage finals | +40 → ~97% | 4,004 |
| genuine failures | 15 timeouts + ~24 absent | |

Fixed in `e3f9af6`: absolute destination pinned in the prompt, stray locations
searched before declaring failure, `run_claude` returns a status string instead
of a bool, and `skills_recovered` / `skills_failed` record per-skill reasons.

- [ ] **Extend `recover_stray_curation_csvs.py` to the stage finals.** Each skill
      keeps its own scratch (`.pk_curation_scratch`, `.pe_study_outcome_scratch`,
      …) with a numbered final (`13_final.csv`, `05_final.csv`, …) plus a
      `combined_final.csv` when several tables were curated. Prefer the combined
      file; otherwise concatenate the per-table finals **only when every part's
      header is byte-identical** — 3 cases in this batch fail that check and must
      be refused, not glued. Verified for `pk-summary-curation` that
      `13_final.csv` carries the same header as a delivered CSV; **the other nine
      are unverified.** Mark these `recovered:stage`, since the skill's final step
      may do more than concatenate.
- [ ] **15 timeouts at the 3600 s cap**, concentrated in `pe-study-outcome` — the
      pipeline handling the largest tables. Genuine failures; needs a longer cap
      or table-level chunking.
- [ ] **Zero-row CSVs are ambiguous.** A skill signals "nothing to extract" by
      writing a header-only CSV, but so does one that gave up. 43 of 678
      delivered CSVs are header-only.

## Open

### 1. Design-stage ambiguity — the real remaining problem

**45 of 199 papers still produce different pipeline sets with all randomness
removed.** At temperature 0 a paper that still flips is flipping because the
prompt is ambiguous, not because the sampler rolled differently.

- **34 of the 45** have a perfectly stable `identify` label — the instability has
  moved almost entirely into Stage 2 design. Only 11 still vary at identify.
- Jaccard 0.946 means these are near-misses: a typical unstable paper flips **one**
  pipeline while agreeing on the rest. "26% unstable" overstates the damage.

Pipelines that flip on/off across repeats:

```
19  pk-population-summary        9  pe-study-outcome
13  pk-specimen-summary          8  pk-population-individual
12  pk-individual-curation       7  pk-summary-curation / pe-study-info
11  pk-drug-individual           4  pk-specimen-individual
10  pk-drug-summary
```

That clusters onto three underspecified rules in `02_design.md`:

- [ ] **Define single-drug and single-specimen explicitly.** Drives the
      `pk-drug-*` (21) and `pk-specimen-*` (17) flips. Replace the plasma-only
      special case in the "Stability rule" with a general rule.
- [ ] **Extend the granularity rule to prose.** State that a range or mean across
      subjects is summary-level. Drives `pk-population-summary` (19) and
      `pk-individual-curation` (12) — the summary-vs-individual call.
- [ ] Pull the actual disagreeing papers for these pipelines so the edits are
      grounded in real cases rather than a reading of the prompt.

### 2. Smaller items

- [ ] **~2% of calls skip Stage 2 entirely** — no `design.json` at all, distinct
      from the schema issue. They still write a correct `selected_pipelines.json`,
      so this is an auditability gap, not a correctness one. (5 further cases are
      `identify = Neither`, where skipping Stage 2 is what the prompt *requires* —
      the scorer should exclude those by design.)
- [ ] **11 papers still vary at `identify`** — worth reading individually; at
      temperature 0 these are genuinely borderline or a genuine prompt gap.
- [ ] **`presence_penalty 1.5`** is inherited from the base model by
      `qwen3.6:35b-t0`. Suspicious for a task that must repeat near-identical
      pipeline names. A `t0-pp0` arm was proposed and never run — low priority
      now that greedy alone reached 94%.
- [ ] **Adopt temperature 0 for production routing**, not just for experiments.
      Nothing currently pins it outside the stability harness.
- [ ] **No ground truth exists for these papers.** Every number above is
      self-consistency. Judging correctness needs reference labels; a Claude
      labelling pilot was scoped and abandoned.

## Branch state

`wip/fengsh/fix-route-issues` — 9 commits, **none pushed**. `main` is ~6 months
stale and merging would drag 79 unrelated commits, so where these land is still
undecided.

The temperature-0 work (Modelfile, slurm scripts, scorers, run artifacts) lives
in the scratch working directory and is **not** in the repo. The only repo
changes from this investigation are `fb353c0` (identify bullet-3) and the
`02_design.md` schema pin.
