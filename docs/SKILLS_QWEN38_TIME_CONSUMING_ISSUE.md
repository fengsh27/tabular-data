# Skills mode + qwen3.8: time-consuming issue — analysis and current status

**Status: root-caused, fixed, validated in production. One side-effect (2 papers needing a
retry) is documented but not fully explained — see "Open question" below.**

This is a standalone write-up of the investigation also summarized in
`docs/TODO_QWEN38_ISSUES.md`'s second section. Scope here is specifically *why skills mode +
qwen3.8 was so slow* and *how it was fixed* — the unrelated pipeline/multi-agent table-selection
bug in that file's first section is a different issue.

## Symptom

Running the `pk-individual-curation` Claude Skill with `qwen3.8:27b-t0` (via Ollama's
Anthropic-compatible endpoint) took far longer per paper than any other mode/model combination
in this benchmark:

| | per-paper time | total (9 papers) |
|---|---|---|
| Original (before fix) | 62–174 min | **16h47m**, 48.3M tokens |
| Pipeline/multi-agent qwen3.8, same model | 4–26 min | 1.2–1.8h |

The skills-mode run wasn't just slower in proportion to doing more work — it was the single
most expensive arm in the entire pk-individual benchmark by a wide margin, on both time and
tokens.

## Root cause

### 1. It is not a model or task-complexity problem

Each paper runs the same `pk-individual-curation` skill: 17 numbered steps
(`00b_select_pk_tables.md` through `14_verify_and_correct.md`) executed as turns inside **one
continuous Claude Code conversation**. Pipeline/multi-agent mode do the equivalent work via
**fresh, independent LangChain prompts per step** — no shared conversation, so step 12's prompt
is the same size as step 1's no matter how many steps have run. Skills mode's conversation,
by contrast, resends everything that came before on every single call.

That architecture alone would make skills mode *more expensive* than pipeline/multi-agent (more
tokens resent per call as the conversation grows), but it does not by itself explain a
10x+ wall-clock blowup — a well-functioning prefix cache should mean each call only pays for
the *new* tokens since the last call, not the whole history.

### 2. Ollama's prefix cache exists and is enabled — but was never actually hitting

Confirmed directly from a completed run's server log
(`load_model: context checkpoints enabled, max = 32, min spacing = 8192`): llama.cpp's
context-checkpoint caching mechanism is on by default. Tracing every call of a full 34-turn,
~104K-token conversation (PMID 34746508):

- Call 1 creates a checkpoint at token position **14,624**.
- **Every one of the next 33 calls** restored *exactly* that same checkpoint — never anything
  larger — then immediately `erased invalidated context checkpoint` for the two newer
  checkpoints it had created after the *previous* call.
- This held to the very end: at 103,792 total tokens, the cache still only recovered the
  first 14,624 — meaning **~89,000 tokens were re-prefilled from scratch on that one call
  alone**, and on literally every call after the first.

### 3. The exact byte that breaks the cache

A smoke test with `OLLAMA_DEBUG_LOG_REQUESTS=true` captured consecutive raw request bodies and
diffed them. `system`, `tools` (47KB), `metadata`, and `model` were byte-identical between
calls. The only difference in the shared prefix was `messages[1]`, Claude Code's "Environment"
system-reminder block — **identical text, different JSON shape**:

```json
// call 1
{"role": "system", "content": [
  {"type": "text", "text": "# Environment\n...", "cache_control": {"type": "ephemeral"}}
]}
// call 2 (same text)
{"role": "system", "content": "# Environment\n..."}
```

Claude Code attaches an Anthropic-API prompt-cache breakpoint (`cache_control:
{"type":"ephemeral"}`) to the end of the stable prefix on each call, and moves that breakpoint
forward as the conversation grows (standard behavior, useful against the *real* Anthropic API).
A message that stops being the active breakpoint gets its content collapsed from an
array-with-metadata back to a plain string. This bookkeeping is invisible and meaningless to
Ollama — but it still changes the literal request bytes on *every* call, and llama.cpp's
checkpoint cache does exact-byte prefix matching with zero semantic awareness that the two
forms mean the same thing. Result: the cached prefix is invalidated at that message's position
on every single call, for the whole length of every conversation.

### 4. Would a newer Ollama version fix this?

Checked directly (2026-10-02): there is an exact upstream report of this mechanism,
[ollama/ollama#18431](https://github.com/ollama/ollama/issues/18431) ("system-role messages
inside `messages` are hoisted into the system block, defeating the prefix cache (Claude
Code)"), filed 2026-09-13, with a candidate fix,
[ollama/ollama#18465](https://github.com/ollama/ollama/pull/18465), filed 2026-09-15. **Both
were still open and unmerged as of the fix date.** No released Ollama version (stable ~0.35.0,
nor the 0.35.1-rc0 / 0.40.0-rc0 prereleases) includes it. Upgrading `ollama.sif` would not have
fixed this.

### 5. The real fix lives on the Claude Code side

[anthropics/claude-code#90018](https://github.com/anthropics/claude-code/issues/90018)
documents the identical failure pattern independently, with another user's own before/after
metrics:

| | input tokens | cache read |
|---|---|---|
| default (`padded-countdown`) | 60k → 77k, growing | stuck at a fixed floor, every call |
| `"totalTokensReminder": "off"` | drops sharply | grows incrementally, as expected |

The message we caught changing shape is exactly this feature: Claude Code's "N tokens left"
system-reminder, re-injected near the start of every request. The fix is a one-line Claude
Code client setting — no Ollama change, no skill rewrite, no job-script restructuring.

## The fix

Added to `CLAUDE_CONFIG_DIR/settings.json` before `claude -p` launches (one line in
`jobs/job_skills_qwen38.sh`, right after the directory is created):

```bash
export CLAUDE_CONFIG_DIR="${PROJECT}/.claude_home/${RUN_NAME}/${PMID}"
mkdir -p "$CLAUDE_CONFIG_DIR"
echo '{"totalTokensReminder": "off"}' > "${CLAUDE_CONFIG_DIR}/settings.json"
```

### Confirmed before trusting it

An A/B smoke test (`jobs/job_skills_qwen38_smoke_debug.sh` vs
`jobs/job_skills_qwen38_smoke_noreminder.sh`, same paper, `OLLAMA_DEBUG_LOG_REQUESTS=true`
in both) showed:

- **Before**: `erased invalidated context checkpoint` on every single call (2 events/call).
- **After**: **zero** such events across 7 captured calls; each call's cache now starts near
  the *end* of the previous call instead of resetting to a fixed early position:

  | call | total tokens | cache starts at | tokens actually reprocessed |
  |---|---|---|---|
  | 2 | 20,176 | 15,376 | 4,800 |
  | 3 | 21,166 | 20,642 | **524** |
  | 4 | 22,723 | 21,904 | 819 |
  | 5 | 24,478 | 23,194 | 1,284 |
  | 6 | 29,682 | 24,587 | 5,095 |
  | 7 | 35,500 | 30,947 | 4,553 |

Reprocessing cost now tracks the size of what's actually new since the last call, not the
entire accumulated conversation.

## Production rollout and results

Applied to the production `job_skills_qwen38.sh`; all 9 pk-individual papers re-run.

| | before | after | change |
|---|---|---|---|
| Total time (9 papers) | 16h47m | 4h56m | **~3.4x faster** |
| Total tokens | 48.3M | 31.9M | ~1.5x fewer |
| Mean semantic score | 89.1 | 86.0 | -3.1 |
| F1 strict macro | 0.900 | 0.910 | +0.010 |

Quality held (one metric down slightly, the other up slightly) — this remains the best-F1 arm
in the entire pk-individual benchmark. Per-paper speedups on the 7 papers that succeeded on
the first attempt ranged **2.3x–4.4x**.

### A separate issue surfaced during rollout

2 of the 9 papers (23200982, 34746508) **failed on the first production attempt** with a
different, likely unrelated failure signature: the model produced an extremely long,
unproductive "thinking" block (77K–282K characters) with no tool call and no visible output,
eventually either stalling silently (Claude Code's "please produce a user-visible response"
nudge, twice, then the session just ended) or exceeding the 64,000-token output cap
(`API Error: Claude's response exceeded the 64000 output token maximum`). Both were cleanly
produced before the fix, on the exact same papers, so this is not a pre-existing weakness of
those papers — it appeared alongside the reminder fix.

**Both succeeded on a simple retry** (same settings, no other changes) — 21.3 min and 43.9 min
respectively, with clean output and (for 23200982) a verification pass with zero findings.

## Open question (not resolved)

`qwen3.8:27b-t0` is configured for greedy, deterministic decoding (temperature 0, fixed seed).
A retry succeeding where the first attempt failed, with no other change, means either:

- genuine GPU/scheduling nondeterminism (e.g. under speculative/MTP draft decoding) breaking
  the theoretical determinism, or
- something about removing the token-reminder text changed the model's effective "pacing"
  signal just enough to occasionally let it spiral into unbounded thinking, with retries
  simply reducing the odds of landing on the bad trajectory again, or
- an unrelated, pre-existing flakiness that the earlier (slower) run happened not to hit on
  these two papers.

**Not investigated further.** If `skills_qwen38` is re-run again and this recurs, it's worth
checking whether it correlates with a specific paper/table shape, or whether it's purely
intermittent. No code or prompt change has been made to address it — the production workaround
today is "retry the failed paper once."

## What this does not fix

The underlying single-conversation architecture is unchanged: skills mode still resends the
whole growing conversation every call, and tokens/time still scale with conversation length,
just without the cache-defeating multiplier on top. It remains the most expensive mode in this
benchmark by both time and tokens (just no longer a *dramatic* outlier). The deeper
architectural fix considered and not pursued — splitting the skill's 17 steps into independent,
narrow `claude -p` calls driven by an external script, so per-call cost stays flat regardless
of conversation length — is still documented as a fallback in `docs/TODO_QWEN38_ISSUES.md`, not
implemented, and would need to be applied across all 10 curation pipelines (they share this
architecture) to matter beyond `pk-individual`.

## Related files

- `jobs/job_skills_qwen38.sh` — production job, now with the fix (on scratch, not this repo)
- `jobs/job_skills_qwen38_smoke_debug.sh` / `job_skills_qwen38_smoke_noreminder.sh` — the A/B
  smoke-test harness used to confirm the mechanism and the fix (on scratch, not this repo)
- `docs/TODO_QWEN38_ISSUES.md` — the fuller write-up, including the unrelated
  pipeline/multi-agent table-selection bug and the originally-proposed (now superseded as the
  primary plan) external-per-step-driver rewrite
- `benchmark_results/20260919/README.md` (on scratch) — caveat 18, and the scores/time/token
  tables reflecting the fixed, final `skills_qwen38` run
- Upstream reports: [ollama/ollama#18431](https://github.com/ollama/ollama/issues/18431),
  [ollama/ollama#18465](https://github.com/ollama/ollama/pull/18465),
  [anthropics/claude-code#90018](https://github.com/anthropics/claude-code/issues/90018)
- Commits: `4243436` (original diagnosis + stale data), `16d3dc0` (fix applied, production
  re-run, re-scored data)
