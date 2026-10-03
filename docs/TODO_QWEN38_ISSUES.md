# qwen3.8 pipeline/multi-agent: PMID 33253437 silently lost (TODO, not fixed)

Status: **diagnosed, not implemented.** Written up on request after a user question
("why did qwen3.8 fail to find a table for 33253437") during the 2026-10-01 pk-individual
benchmark addition; see `benchmark_results/20260919/README.md` caveat 17 for the one-line
summary and the score impact. This file is the full root-cause trace and the fix plan.

## Symptom

`pipeline_qwen38` and `multiagent_qwen38` (both `PIPELINE_LLM=AGENT_LLM=QWEN3.8-27B`) each
produced output for only 8 of the 9 pk-individual benchmark papers. The missing one,
PMID 33253437, ends with:

```
2026-10-01 12:11:50,312 - extractor.agents_manager.pk_pe_manager - INFO - Result (drug_list):
2026-10-01 12:11:50,312 - extractor.agents_manager.pk_pe_manager - INFO -
2026-10-01 12:11:50,321 - scripts - ERROR - No curated table found for 33253437 pk_individual
```

No exception, no `ERROR`-level log line anywhere upstream of that final message. Every
other model tested on this paper (gpt-4o, gpt-5.4, qwen3.6) succeeds
(`benchmark_results/20260919/README.md`'s per-paper tables: semantic 72–90, F1 0.38–0.91
depending on run). The "no curated table found" phrasing is misleading about *where*
things went wrong — table selection actually succeeded.

## Root cause, traced step by step

Full trace: `logs/pipeline_qwen38/33253437.log` (pk-individual benchmark working directory,
`/fs/scratch/PCON0100/feng1426/projects/playground/tabular-data-benchmark/pk-individual/`),
the second run in that file (timestamps from 12:10:41 onward — the first run in the same
file, ending 11:56:20, predates the `instruction_prompt` fix in commit `9524cbd` and failed
earlier, with a different symptom; not relevant here).

**1. Table selection succeeds, with correct reasoning:**

```
Selected tables (indices): ['1']
Reason: Table 0 presents sociodemographic and clinical baseline characteristics (age,
smoking, diagnosis), which are excluded as they do not contain PK data. Table 1 presents
individual pharmacokinetic data, specifically plasma concentrations (ng/ml) for mothers
and infants, and umbilical cord ratios, which directly meet the inclusion criteria for
drug concentration measurements and PK-related ratios.
```

**2. Preprocessing, drug extraction, population extraction, population refinement, and the
"Deleting Summary Data" step all run correctly.** The model extracts 6 drugs (Sertraline,
Citalopram, Escitalopram, Venlafaxine, Paroxetine, Fluoxetine), correctly identifies and
removes a trailing summary row (row index 33), and the resulting `md_table_individual` has
its full original 12 columns intact:

```
| Unnamed: 0 | ID | Dose (mg/d) | Mother's PL III trimester (ng/ml) | Mother's PL (ng/ml)
delivery | Infant's PL (ng/ml) | Umbilical maternal ratio (%) | Other drugs | Expected
phenotype (CYP2D6: major CYP metabolizer) | Maternal outcomes | Bleeding (ml) | Neonatal
outcomes |
```

**3. "Aligning Parameter Type" is where it breaks.** This step
(`extractor/agents/pk_individual/pk_ind_param_type_align_step.py` /
`pk_ind_param_type_align_agent.py`) asks the model one classification question: is the PK
parameter type (Cmax, concentration, etc.) represented as column headers already (wide
format), or as repeating values inside one particular column (long format, needs
reshaping)? The prompt (`PARAMETER_TYPE_ALIGN_PROMPT`):

> "col_name is the column name \[...] it represents the PK parameter type \[...] serves as
> the row header or is listed under the specific column. If the PK parameter type is
> represented as column headers, use null."

qwen3.8 answered:

```
{"col_name": "Unnamed: 0"}
```

`Unnamed: 0` is the row-index/patient-ID column (values 1, 3, 4, 5, ..., 33) — not a column
of parameter-type labels. This is a model reasoning error on this specific table shape (a
wide table whose first column is an unlabeled index). For comparison, on the *same paper,
same table*, both gpt-4o and gpt-5.4 answer `col_name: null` (correctly: the table is
already wide) and the 12-column table passes through untouched:

```
== pipeline_gpt4o ==
Result (md_table_aligned):
| Unnamed: 0 | ID | Dose (mg/d) | Mother's PL III trimester (ng/ml) | ... | Neonatal outcomes |
| Sertraline | 1 | 75 | 19.5 | NA | NA | NA | - | UM | - | 250 | Apnea with desaturation ... |

== pipeline_gpt54 ==
(identical)

== pipeline_qwen38 ==
Result (md_table_aligned):
| Unnamed: 0 | Escitalopram |
| ID |  |
```

**4. The code acts on the bad answer without validating it, and destroys the table.**
`post_process_parameter_type_align` (`pk_ind_param_type_align_agent.py:60-69`):

```python
if res.col_name is None:
    return dataframe_to_markdown(df_table)          # no-op: already wide
else:
    df_table = f_transpose(df_table)                  # unconditional, whole-table transpose
    return deduplicate_headers(
        fill_empty_headers(fill_empty_headers(remove_empty_col_row(dataframe_to_markdown(df_table))))
    )
```

Two defects here, not one:

- **`res.col_name` is never validated or even used.** `fix_col_name` is imported
  specifically to fuzzy-match a model-given string against the table's real column names,
  but the call to it is commented out (lines 56–59, 61–62 — leftover from an earlier
  version). The live branch transposes *regardless of which column was named or whether it
  could plausibly hold parameter-type labels*, and never renames anything to
  `"Parameter type"` afterward (that assignment is also commented out, line 66) — so even a
  *correct* column name would not currently produce a column downstream code can find by
  that name.
- **`f_transpose` is the wrong operation for the question being asked.** A "yes, column X
  holds repeating parameter labels" answer calls for pivoting that column's values into new
  columns (long → wide), not swapping every row and column in the table. `f_transpose` does
  a full transpose. On a wide table (one row per patient, 12 columns of mixed types), a full
  transpose using the ID column's values as new headers is maximally destructive: patient
  IDs (1, 3, 4, ..., 33) become column headers, the original 12 column names become row
  values, and after `deduplicate_headers`/`fill_empty_headers`/`remove_empty_col_row` run on
  that wreckage, only 2 columns survive (`Unnamed: 0`, `Escitalopram` — the latter is one of
  the original row *values*, not a real header). This branch was apparently never exercised
  correctly before: gpt-4o/gpt-5.4 always take the `None` branch on every paper tested so
  far, so the transpose path's breakage was latent until a model finally answered non-null.

**5. Every step after that fails silently, because nothing checks the table is still
usable** — there is no exception anywhere in this chain, each step just has nothing to find:

| step | result | why |
|---|---|---|
| Categorizing Column Header | `{'Unnamed: 0': 'Patient ID', 'Escitalopram': 'Uncategorized'}` | only 2 columns exist; the one data column can't be identified as a value column |
| Sub-table Creation | `[]` | no `"Parameter value"` column exists to split on |
| Parameter Type/Unit/Value Extraction | `[]` | cascades from the empty sub-table list |
| Drug Matching (Agent) | `[]` | cascades further |
| (final assembly) | `NoTableFoundError` → `"No curated table found for 33253437 pk_individual"` | nothing was ever produced |

This is the identical *failure shape* to a bug already found and fixed in the pk-summary
pipeline (commit `97fb1cb`, see `tests/test_pk_sum_header_categorize_value_column.py`'s
docstring): one upstream model misclassification, with no validation anywhere on the path,
silently empties the whole table downstream with no error raised. That fix added a guard at
the header-categorization step specifically for pk-summary
(`find_unlabeled_value_columns` / `try_fix_error_header_categories` in
`extractor/agents/pk_summary/pk_sum_header_categorize_agent.py`). **The equivalent guard
does not exist for pk-individual's header-categorize step
(`pk_ind_header_categorize_agent.py`)**, and the align-step bug described above is further
upstream of that than the pk-summary case, inside a different, more destructive operation
(a full-table transpose rather than a labeling choice).

## Why it only showed up now

`multiagent_qwen38` and `pipeline_qwen38` are the first time `QWEN3.8-27B` has been run
through `PKPEManager`'s pipeline/multi-agent path at all (added in commit `a1f980d`, this
benchmark addition) — every earlier pk-individual run used gpt-4o, gpt-5.4, qwen3.6 or
gemma4 here, none of which answered non-null on any paper tested. qwen3.8 in *simple-prompt*
or *skills* mode never goes through this step (those modes don't use `PKIndWorkflow`'s
agent chain at all), so this is also not something either of qwen3.8's other two arms in
this benchmark could have surfaced.

## Why the skills and simple-prompt qwen3.8 arms don't have this failure

Confirmed from the scores, not just architecture reading: on this exact paper,
`simple_qwen38` scores 82.0/0.912 and `skills_qwen38_aug26` scores 89.0/0.912 - qwen3.8
itself has no trouble with this table's content. Both are at `benchmark_results/20260919/scores_matrix.csv` and `f1_matrix_strict.csv`.

**`skills_qwen38_aug26`** runs `ollama_skills/pk-individual-curation/prompts/05_param_type_align.md`,
a direct prose port of this same step, at the same point in the same conceptual pipeline -
so this isn't a different task decomposition, it's a port that happens to fix, by
construction, the two things broken in the code version:

1. **The skill has the model produce and inspect the actual resulting table, not a one-token
   signal a separate code path blindly acts on.** The LangChain step asks for
   `{"col_name": "..."}`, completely divorced from the table it will be applied to, and
   `post_process_parameter_type_align` mechanically transposes the real table based on
   nothing else. The skill instead has the model reason about the case and then *write out
   the resulting table itself*. A model trying to actually carry out a transpose using
   patient IDs as new column headers, in full view of the real content, is far more likely
   to notice that's nonsense than a model asked to emit one isolated classification token.
2. **The skill's prompt has an explicit self-check-and-retry instruction that the code has no
   equivalent of**: "Before continuing, sanity-check: each data row corresponds to a single
   subject, the parameter types appear as column headers (not down a column), no value cells
   were lost, merged, or reformatted. **If that check fails, redo this stage once.**" This is
   exactly Layer 2 below, already present as a prompt-level instruction - existing proof that
   a self-check-and-retry step catches this failure in practice, for the identical model on
   the identical kind of table. The code path's only retry trigger is a schema/parse failure,
   never a semantic/structural one.
3. **The skill's prompt includes a worked example matching this paper's exact shape** - an
   `ID` column plus named parameter-unit headers, labeled "Case A - the common case, keep
   as-is." The LangChain prompt (`PARAMETER_TYPE_ALIGN_PROMPT`) has no worked example, just an
   abstract description, plausibly making an ambiguous-looking index column easier to
   misjudge.

**`simple_qwen38`** sidesteps the question entirely: `prompts/simple_prompts/pk_individual.md`
is a single flat prompt asking the model to write the final 12-column CSV rows directly off
the raw table. There is no orientation-detection step, no transpose, no intermediate state
machine - nothing here for a model to get wrong.

## Proposed fix (three layers, not yet implemented)

### Layer 1 — stop trusting an unvalidated column name (the direct fix)

In `post_process_parameter_type_align`, before acting on a non-null `res.col_name`:

1. Resolve it against the table's real columns with the already-imported `fix_col_name`
   (currently dead code in this function).
2. Reject it if it names a column already known to be a per-row identifier — an unlabeled
   index column, or a header matching the same vocabulary the header-categorize step
   already uses to recognize `"Patient ID"` (`"Unnamed: 0"`, `"ID"`, `"Patient ID"`,
   `"Subject"`, `"No."`, case-insensitive). A patient-identifier column can never correctly
   hold parameter-type labels. If rejected, fall through to the same branch as `col_name is
   None` — a known-bad transpose is strictly worse than a no-op.
3. For a column that passes that check, replace the full-table `f_transpose` with an actual
   pivot: the named column's distinct values become new columns, and the row's measurement
   values populate them — not a row/column swap of the whole table. Nothing in this
   codebase currently exercises a genuinely long-format pk-individual table correctly
   (every real paper seen so far is wide), so this branch should be validated against a
   real long-format fixture, not just unit-tested against made-up data, before being
   trusted.

### Layer 2 — a structural sanity check, in case layer 1's heuristic misses a case

Mirroring `extractor/agents/pk_summary/pk_sum_header_categorize_agent.py`'s
`RetryException` pattern: after the transpose decision, check whether the result is
structurally plausible (e.g., column count dropped by more than some threshold relative to
input, or no remaining column looks like it could hold numeric/parameter data). If not,
raise a `RetryException` naming the problem ("transposing on 'Unnamed: 0' left only 2
columns; that looks like a patient identifier, not a repeating parameter-type label —
reconsider") so the agent gets a second attempt with concrete feedback, via the same
retry-with-feedback loop every other step in this pipeline already uses
(`_get_previous_errors_prompt` / `previous_errors`).

### Layer 3 — a generic backstop at header-categorization (highest value for the effort)

Port the pk-summary fix (`find_unlabeled_value_columns` +
`try_fix_error_header_categories`, `extractor/agents/pk_summary/pk_sum_header_categorize_agent.py`)
to pk-individual's own header-categorize step/agent
(`pk_ind_header_categorize_agent.py` / `pk_ind_header_categorize_step.py`). This is the
single highest-value change of the three: it catches this bug *and* any other future bug
with the same failure shape — anything upstream that leaves every column "Uncategorized"
and no `"Parameter value"` column — at the one choke point right before
`SplitByColumnsStep` silently returns `[]`. Layers 1 and 2 are specific to this one
mis-answered question; layer 3 is a backstop regardless of cause.

## Suggested tests (follow `tests/test_pk_sum_header_categorize_value_column.py`'s shape)

1. **Unit test, layer 1**: feed `post_process_parameter_type_align` a wide 12-column table
   shaped like 33253437's, with a stub `ParameterTypeAlignResult(col_name="Unnamed: 0")`.
   Assert the table survives unchanged (treated as if `col_name` were `None`), not
   collapsed to 2 columns.
2. **Unit test, layer 3**: mirror
   `test_a_mapping_with_no_value_column_but_numeric_columns_is_sent_back_with_the_columns`
   and `test_fixer_labels_only_the_numeric_columns` from the pk-summary test file, adapted
   to `pk_ind_header_categorize_agent`'s schema.
3. **Regression test, end-to-end**: reproduce the exact cascade (align → categorize →
   split) on 33253437's actual table content (captured from this log) with a stubbed LLM
   that returns qwen3.8's exact recorded answers at each step, asserting a non-empty result
   — so this specific paper's failure mode can't silently return even if the heuristics in
   layers 1–3 are later refactored.

## Not done in this pass

No code was changed for this issue. `pipeline_qwen38` and `multiagent_qwen38` remain 8/9 in
`benchmark_results/20260919/` (see caveats 14 and 17 there). Re-running those two arms after
implementing any of the above has not been scheduled.

---

# Skills mode + qwen3.8: per-paper time blows up because context never resets (TODO, not fixed)

Status: **root-caused, and a one-line fix (`"totalTokensReminder": "off"`) confirmed working
via A/B smoke test - not yet applied to the production job script or measured on a full
paper.** Written up on request
("So, this is skill issue we need to fix. How could we fix it by asking narrow questions
similar to pipeline mode or ma mode?") during the 2026-10-02 re-run of `skills_qwen38` on
pk-individual, prompted by the user not believing the originally-reported 81.8 min total for
`multiagent_qwen38` and then asking why `skills_qwen38` was separately taking so long.

## Symptom

Re-running `skills_qwen38` (9 parallel per-paper Slurm jobs, job script
`jobs/job_skills_qwen38.sh`, `qwen3.8:27b-t0` via the current `ollama.sif`, 196608 context) on
2026-10-02, the 4 papers that finished took 62, 67, 75, and 88 minutes each
(`logs/skills_qwen38/<pmid>.time.json`, `claude_seconds` field); the other 5 were each still
running past 1:42:00 elapsed at last check. For comparison, the *same model* in pipeline or
multi-agent mode (`pipeline_qwen38`, `multiagent_qwen38`) finishes each paper in 4-26 minutes
(see the wave-2 timings in the section above). Skills mode is roughly 3-10x slower per paper
on identical hardware and an identical model.

(An earlier draft of this investigation mis-cited a 217-minute outlier for PMID 23200982 as
evidence - that number came from a stale `.time.json` left over from the original 2-paper
pilot run collected on 2026-09-19 under the same `RUN_NAME=skills_qwen38` directory, re-used
by coincidence of a job ID lookup, not from the 2026-10-02 re-run. The numbers above (62-88
min, confirmed via each `.time.json`'s `"job"` field matching an actual 2026-10-02 Slurm job
ID) are the real ones.)

## Root cause: one single, continuously-growing Claude Code conversation per paper

`skills_qwen38` invokes `claude -p "Use the pk-individual-curation skill..."` **once per
paper**, and the skill then runs all of its steps as turns inside that *same* conversation:

```
ollama_skills/pk-individual-curation/prompts/
  00b_select_pk_tables.md  00c_infer_patient_id.md  01_drug_info.md  02_patient_info.md
  03_patient_refine.md  04_summary_data_del.md  05_param_type_align.md
  06_header_categorize.md  07_split_by_col.md  08_type_unit_value_extract.md
  09_drug_matching.md  10_patient_matching.md  11_time_extraction.md  12_assembly.md
  13_row_cleanup.md  14_verify_and_correct.md
```

17 steps, each potentially several turns (tool calls + responses) — and because it's one
conversation, turn *N* resends every prior turn's prompt and output as part of its own input.
Measured directly from today's session log for PMID 34746508 (the fastest of the 4 that
finished, 62 min total; 34 unique assistant turns, de-duplicated by message id, from
`.claude_home/skills_qwen38/34746508/projects/*/*.jsonl`):

| turn | input tokens | gap since previous turn |
|---|---|---|
| 1 | 15,136 | - |
| 10 | 50,989 | 48s |
| 12 | 66,251 | 81s |
| 20 | 73,805 | 125s |
| 28 | 85,289 | 139s |
| 33 | 89,564 | 134s |

Input size climbs **monotonically, 15K -> 90K tokens**, and per-turn wall time climbs right
alongside it - 13-45s per turn early on, 100-185s per turn later, even on steps that add only
a few hundred tokens of genuinely new content. One single turn took 418s. This is not a
token-accounting artifact (that inflation was already known and documented in `COST_NOTE` in
the figures scripts) - the *wall-clock time* genuinely balloons too, because every call has
to reprocess the entire accumulated context from scratch before it can do anything new. The
per-paper total is the sum of a steadily-increasing per-turn cost across ~30-40 turns, which
is why it lands at 60-90+ minutes for an average paper instead of the few-minutes-per-paper
that 17 individually-small prompts would cost in isolation.

Contributing factor worth separately checking: `OLLAMA_FLASH_ATTENTION=false` in the current
server config. If Ollama/llama.cpp is not reusing KV-cache across calls within the session
(prefix caching), every call pays full prefill cost for the *entire* context, not just the
newly-added suffix - which is consistent with the observed pattern (cost tracks total context
size, not the size of what's new since the last turn).

## Why pipeline and multi-agent mode don't have this problem

Every pipeline/multi-agent step (`pk_ind_param_type_align_agent.py`,
`pk_ind_header_categorize_agent.py`, etc.) builds a **fresh LangChain prompt** and calls the
LLM directly - there is no shared conversation across steps, so step 12's prompt is the same
size as step 1's, regardless of how many steps ran before it. Cost per step is flat; total
cost is (steps) x (flat per-step cost), not a sum of a climbing sequence.

## Proposed fix (not yet implemented): split the one long skill conversation into narrow, independent per-step calls

The goal is to give `pk-individual-curation` the same flat-cost shape pipeline/multi-agent
already have, without giving up on it being a genuine Claude Skill (it still needs to run
under real Claude, not just Ollama). Concretely:

1. **Write an external driver** (bash or Python) that loops over the 17 prompt files in
   order. For each step, it builds a narrow, self-contained prompt — e.g. *"Follow
   `prompts/06_header_categorize.md` for paper `<pmid>`. Input: `<scratch>/05_output.md`.
   Write output to `<scratch>/06_output.md`."* — and invokes `claude -p` as a **brand-new
   process** each time (fresh `CLAUDE_CONFIG_DIR`/session, no `--continue`), so the
   conversation genuinely resets rather than accumulating.
2. **State moves from conversation history to disk, exclusively.** The skill already writes
   scratch files at most steps; the change is to make the file the *only* thing the next
   step's prompt depends on, not "the file plus whatever the model still remembers from
   earlier turns in the chat."
3. **Keep each step's own self-check-and-retry instructions intact, inside that step's own
   prompt** (e.g. `05_param_type_align.md`'s worked example + explicit "if that check fails,
   redo this stage once"). This is what currently protects skills mode from the pipeline-mode
   param-type-align bug documented above, and it survives the split because it's scoped to
   one step, not the whole procedure.
4. **Add a cheap validation gate in the driver, between steps** (does the output file exist,
   is it non-empty, does it parse as a table with a plausible column count) — a natural place
   to also enforce Layer 2/3 of the fix above, catching a bad step's output before it
   cascades into the next one.

This is not a new pattern for this skill suite — `SKILL.md` already dispatches
`prepare-paper` -> `route` -> the chosen curation pipeline as three independent calls. The
fix is to apply that same "independent call per stage" pattern one level deeper, to the 17
steps *inside* `pk-individual-curation` (and, if it works, to the other 9 curation pipelines,
which share the same single-conversation architecture).

**Trade-off to be explicit about:** this is a real re-architecture, not a tuning knob, and it
would need to be repeated across all 10 curation pipelines, not just this one. It also gives
up the one advantage skills mode had (the model seeing its *entire* history let it
self-correct across all 17 steps, not just within one) — after the split, each step only
self-corrects within its own narrow window, moving its failure profile closer to pipeline
mode's for anything that genuinely needs cross-step context to catch.

**Cheaper alternative, investigated 2026-10-02 - ruled out at the config level; root cause
found.** The hope was that Ollama's KV-cache prefix reuse could be made to engage across calls
within one session, letting the existing single-conversation skill keep its self-correction
advantage while only paying prefill cost for the newly-added suffix each turn. A smoke test
(`jobs/job_skills_qwen38_smoke_debug.sh`, `RUN_NAME=skills_qwen38_smoke_debug`, PMID 34746508,
`OLLAMA_DEBUG_LOG_REQUESTS=true`, raw request bodies mirrored off node-local `/tmp` to
`logs/skills_qwen38_smoke_debug/request_logs_34746508/` every 3s since that directory - and
SSH access to the node - disappears the instant the job ends) confirmed the mechanism exists
and is already enabled (`load_model: context checkpoints enabled, max = 32, min spacing =
8192`), but found exactly why it never helps past the first call:

Diffing the first two raw request bodies (`system` 6,240 bytes, `tools` 47,305 bytes,
`metadata`, `model` all byte-identical), the only divergent part of the shared prefix is
`messages[1]`, the "Environment" system-reminder block - same human-readable text both times,
different JSON shape:

```json
// request 1
{"role": "system", "content": [
  {"type": "text", "text": "# Environment\n...", "cache_control": {"type": "ephemeral"}}
]}
// request 2 (identical text)
{"role": "system", "content": "# Environment\n..."}
```

Claude Code attaches an Anthropic-API prompt-cache breakpoint (`cache_control:
{"type":"ephemeral"}`) to the end of the stable prefix on each call and moves that breakpoint
forward as the conversation grows (standard Anthropic SDK behavior, useful against the *real*
Anthropic API); a message that stops being the active breakpoint has its content collapsed
from an array-with-metadata back to a plain string. Against Ollama this bookkeeping is
invisible and meaningless, but it still changes the literal request bytes on every single
call. llama.cpp's context-checkpoint cache does exact-byte prefix matching on the rendered
prompt with no semantic awareness that `[{"type":"text","text":"X","cache_control":{...}}]`
and `"X"` mean the same thing - so the cached prefix is invalidated at this message's position
on literally every call, for the entire length of every conversation. This matches the full
34746508 run exactly: the restored checkpoint was stuck at position 14,624 for all 33 calls
after the first, regardless of how large the conversation grew (confirmed up to 103,792
tokens), forcing a full re-prefill of everything past that point every time.

**This cannot be fixed by tuning Ollama's config** (`OLLAMA_FLASH_ATTENTION`,
`OLLAMA_NUM_PARALLEL`, `OLLAMA_KEEP_ALIVE` are all irrelevant to this) - the server-side cache
is already working correctly by its own logic; it is being fed a prefix that genuinely is not
byte-stable, through no fault of Ollama's. Checked whether a newer Ollama version fixes it at
the server side: there is an exact upstream report of this same mechanism,
[ollama/ollama#18431](https://github.com/ollama/ollama/issues/18431) ("system-role messages
inside `messages` are hoisted into the system block, defeating the prefix cache (Claude
Code)"), with a candidate fix,
[PR #18465](https://github.com/ollama/ollama/pull/18465) - but as of 2026-10-02 both are still
open and unmerged (filed 2026-09-13/15; checked live), and no released version (stable ~0.35.0,
nor the 0.35.1-rc0/0.40.0-rc0 prereleases) includes it. Upgrading `ollama.sif` would not have
helped today.

**Fix found and confirmed: `{"totalTokensReminder": "off"}` in Claude Code's
`settings.json`.** The message we caught changing shape (array-with-`cache_control` vs plain
string) is Claude Code's own "N tokens left" system-reminder block, re-injected near the start
of every request. This is independently documented as the same class of bug on the Claude Code
side - [anthropics/claude-code#90018](https://github.com/anthropics/claude-code/issues/90018),
"totalTokensReminder causes repeatable prompt-cache floor in tool loops; off restores
incremental hits" - with another user's before/after metrics showing the identical signature
(cache reads frozen at a fixed floor regardless of growing input tokens, until the setting is
turned off). **Confirmed directly against this project's own setup, not just by analogy:**
`jobs/job_skills_qwen38_smoke_noreminder.sh` (same smoke-test harness as
`job_skills_qwen38_smoke_debug.sh`, `OLLAMA_DEBUG_LOG_REQUESTS=true` still on, plus
`echo '{"totalTokensReminder": "off"}' > "$CLAUDE_CONFIG_DIR/settings.json"` before launching
Claude Code) ran the same paper (34746508) and produced zero `erased invalidated context
checkpoint` events across 7 captured calls (vs. 2 on literally every call in the broken run),
with each call's cache now starting near the *end* of the previous call instead of resetting
to a fixed early position:

| call | total tokens | cache starts at | tokens actually reprocessed |
|---|---|---|---|
| 2 | 20,176 | 15,376 | 4,800 |
| 3 | 21,166 | 20,642 | 524 |
| 4 | 22,723 | 21,904 | 819 |
| 5 | 24,478 | 23,194 | 1,284 |
| 6 | 29,682 | 24,587 | 5,095 |
| 7 | 35,500 | 30,947 | 4,553 |

Reprocessing cost now tracks the size of what's actually new since the last call (hundreds to
a few thousand tokens), not the entire accumulated conversation - exactly the flat, incremental
cost pipeline/multi-agent mode already has, achieved here with a one-line client-side setting
change and zero changes to Ollama, the skill, or the job script's core logic.

**Not yet done:** apply `{"totalTokensReminder": "off"}` to the production
`jobs/job_skills_qwen38.sh` (and the other skills-mode job scripts, `job_skills_qwen36.sh` /
`job_skills_gemma4.sh`, which likely have the same bug) and re-run a full paper to measure the
actual wall-clock/token savings end to end - the smoke test above only confirms the caching
mechanism now works, not the final per-paper time this yields. Given this fix is this cheap and
this well-confirmed, it should be tried in production before investing in the external
per-step-driver rewrite, which remains the fallback if this doesn't fully resolve it (e.g. if
some other message later in a long conversation turns out to have the same instability).

## Not done in this pass

The production job scripts have not been updated with the `totalTokensReminder: off` setting
yet, and no full-paper run has been re-measured with it. The 2026-10-02 `skills_qwen38` re-run
(9 parallel jobs, without this fix) completed and was scored (see
`benchmark_results/20260919/README.md`) before this fix was found. The smoke-test artifacts
(`logs/skills_qwen38_smoke_debug/`, `logs/skills_qwen38_smoke_noreminder/`,
`output/skills_qwen38_smoke_debug/`, `output/skills_qwen38_smoke_noreminder/`,
`.claude_home/skills_qwen38_smoke_debug/`, `.claude_home/skills_qwen38_smoke_noreminder/`,
`skills_work/skills_qwen38_smoke_debug/`, `skills_work/skills_qwen38_smoke_noreminder/`) are
left on scratch for reference and are not part of the scored benchmark data.
