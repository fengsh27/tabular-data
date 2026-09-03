# simple-prompt experiment (pk-individual)

Measures the fast path of the two-tier strategy: **one model call per table**,
no decomposition, no verification — against `ollama_skills`' 15-stage pipeline.
Both arms read the same tables, so the only variable is the prompt.

## Files

| file | role |
|---|---|
| `_common.py` | parses model output into the canonical 13-column schema. Strips `<think>` blocks and code fences, maps every spelling of *missing* to `""`, and melts wide `parameter1/value1/...` output to long. |
| `run_simple_prompt.py` | renders the prompt per table and calls Ollama. |
| `score_pk_individual.py` | scores predictions against the manual gold. No LLM, no network. |
| `aliases.py` | vocabulary equivalences, used only with `--aliases`. |

## Run

```bash
# fast path, v2 prompt
python scripts/simple_prompt/run_simple_prompt.py \
    --prompt prompts/simple_prompts/pk_individual.md \
    --scratch /path/to/.pk_individual_scratch \
    --out results/simple_v2 \
    --model qwen3.8:27b --temperature 0 --resume

python scripts/simple_prompt/score_pk_individual.py \
    --pred results/simple_v2 --label simple-v2 --aliases safe
```

`--dry-run` renders the prompts without contacting the model. `--resume` reuses
saved raw responses, so a re-score costs nothing.

The runner reads `<scratch>/<pmid>/table_*/00_markdown_table.md` — the tables the
skill actually selected — falling back to `inputs.md`. Point `--scratch` at a
completed skill run so both arms see identical input.

## Scoring

Two views, because they answer different questions.

- **values** — bag-of-numbers recall/precision. Ignores row identity, so it only
  says whether the right numbers were read off the table. Always flattering;
  treat it as an upper bound.
- **rows** — one-to-one row matching (equal `Parameter value` required), then
  per-column accuracy over matched pairs. This is the number that reflects
  whether the schema was filled in correctly.

Always normalised, because it is spelling rather than meaning:

- micro sign — `µg/L`, `μg/L`, and `ug/l` are one unit
- case and whitespace

Behind `--aliases`, because it is a judgement call:

- `safe` — the gold's wording vs the controlled vocabulary the legacy extractor
  emits (`delivery` ≡ `Parturition/Labor/Delivery`, `lactation` ≡
  `Nursing/Breastfeeding/Lactation`, `cord blood` ≡ `umbilical cord blood`, ...)
- `all` — additionally the debatable subject mappings (`maternal` ≡ `Adults`,
  `pediatric` ≡ `Infants`). Review `aliases.py` before trusting these.

`--confusions "Parameter type,Pregnancy stage"` dumps the top gold→predicted
disagreements per column. Run this before believing any column's score — the
first pass revealed 166 rows failing only on the micro sign.

## Verify the scorer

Scoring the gold against itself must be 100% everywhere:

```bash
for f in benchmark/data/pk-individual/baseline/*_baseline_manual.csv; do
  p=$(basename "$f" | cut -d_ -f1); mkdir -p /tmp/sc/$p; cp "$f" /tmp/sc/$p/combined.csv
done
python scripts/simple_prompt/score_pk_individual.py --pred /tmp/sc --label self-check
```

## Baseline: ollama_skills + qwen3.8:27b

9 papers (32635742 has no manual gold), 725 predicted vs 664 gold rows.

```
values   recall 96.0%   precision 74.6%
rows     recall 81.5%   precision 74.6%

per-column accuracy over 541 matched rows      off     safe      all
  Patient ID                                  99.1%   99.1%   99.1%
  Drug name                                   98.5%   98.5%   98.5%
  Time value / Time unit                      96.9%   96.9%   96.9%
  Analyte                                     78.0%   78.0%   78.0%
  Parameter unit                              76.2%   75.8%   75.8%
  Specimen                                    45.7%   68.9%   68.9%
  Pediatric/Gestational age                   57.1%   57.1%   57.1%
  Population                                  40.7%   40.7%   60.1%
  Pregnancy stage                              9.6%   53.0%   53.0%
  Parameter type                              17.6%   22.7%   22.7%
```

Reading: the model reliably gets *which patient*, *which drug*, *which number*.
It is unreliable on every descriptive column, and `Parameter type` stays worst
(22.7%) even after all vocabulary allowances — that is a real defect, not a
scoring artifact.
