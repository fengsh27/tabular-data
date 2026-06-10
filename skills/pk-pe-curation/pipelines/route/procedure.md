# PK/PE Route

The **selector** of the curation-skills suite. It ports two legacy steps —
`PKPEIdentificationStep` (PK/PE/Both/Neither) and `PKPEDesignStep` (multi-label
pipeline selection) — and emits a deterministic dispatch artifact. It reads the
output of `prepare-paper`; run that first.

This skill **does not curate**. Its job is to answer "which of the 10 curation
skills should run on this paper?" and hand you that list.

## Inputs you need
The prepared-paper directory from `prepare-paper` (`./.paper_assets/<pmid>/`):
- `paper_text.md` — title (H1) + body with `[Table N]` markers.
- `abstract.md` — used by Stage 1.
- `table_<n>.md` / `table_<n>.html` — spliced back into the full text for Stage 2.
- `manifest.json` — the `[Table N]` ↔ `table_<n>.*` index.

If the user has not run `prepare-paper` yet, run it first (or, if they paste raw
title/abstract/full text, you can proceed with that text directly — the file
layout is just the convenient form).

## Scratch directory
Write intermediate files to `./.pk_pe_route_scratch/<pmid>/` in the user's current
project / working directory (never inside the skill folder; it is git-ignored):
`identify.json`, `design.json`, and the final `selected_pipelines.json`.

## Procedure
1. **Stage 1 — Identify** (`prompts/01_identify.md`): from title + `abstract.md`,
   classify the paper as PK / PE / Both / Neither → `identify.json`. **If
   `Neither`, stop**: write an empty `selected_pipelines.json` and report that the
   paper is out of scope.
2. **Stage 2 — Design** (`prompts/02_design.md`): reconstruct the full text with
   tables visible (splice each `table_<n>.md` at its `[Table N]` marker), then
   select the **union** of all applicable pipelines (multi-label, non-exclusive) →
   `design.json`.
3. **Deterministic dispatch**: map the selected pipeline labels to skill folders
   with the bundled, byte-tested table — never hand-write the skill names:
   ```bash
   python pipelines/route/scripts/pipeline_skill_map.py \
       --pmid <pmid> --paper-type <PK|PE|Both> \
       <pipeline_tools from design.json> > ./.pk_pe_route_scratch/<pmid>/selected_pipelines.json
   ```

## Output — `selected_pipelines.json`
```json
{ "pmid": "12345678", "paper_type": "Both",
  "selected": [
    {"pipeline": "pk_summary", "procedure": "pipelines/pk-summary-curation"},
    {"pipeline": "pe_study_outcome", "procedure": "pipelines/pe-study-outcome"} ] }
```

Each `procedure` is a bundle-relative directory; the orchestrator
(`SKILL.md`) follows `<procedure>/procedure.md` for each selected pipeline. Hand
this list back to the orchestrator — by default it reports the selection and lets
the user confirm before running each pipeline (the robust choice under smaller
Ollama models); full end-to-end dispatch is the opt-in path described in `SKILL.md`.

## Notes
- **Tables are visible to the design stage by design.** The legacy step saw table
  data inline in the full text; Stage 2 reconstructs that by splicing the
  `table_<n>.md` files back at their markers. Do not run the design stage on the
  bare `paper_text.md` (markers only) — it would see less than the legacy step.
- The two model stages are evaluated against semantic oracles by hand / eval
  harness, like every other model stage. The **label→procedure map is
  deterministic** and guarded by `skills_e2e_tests/test_pipeline_skill_map.py`.
