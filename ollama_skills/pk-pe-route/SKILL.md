---
name: pk-pe-route
description: Decide which PK/PE curation pipelines apply to a paper. First classifies the paper as PK / PE / Both / Neither from its title + abstract, then selects the matching pipelines — pk_* for a PK paper, pe_* for a PE paper, both for Both, and none for Neither — and writes the selected pipeline skills to trigger next. Use after pk-pe-prepare, when the user asks which pipelines to run on a paper. Does NOT curate; returns an empty selection for non-PK/PE papers.
---

# PK/PE Route

The **selector** of the curation suite. It ports two legacy steps —
`PKPEIdentificationStep` (PK / PE / Both / Neither) and `PKPEDesignStep`
(multi-label pipeline selection) — and writes the list of curation skills that
apply to the paper. It **does not curate**: hand the selection to the matching
pipeline skills (or trigger them yourself), one at a time.

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

## Prerequisite
Run **pk-pe-prepare** first to produce `./.paper_assets/<pmid>/` (`paper_text.md`,
`abstract.md`, `table_<n>.md` / `table_<n>.html`, `manifest.json`). If the user
pasted raw title / abstract / full text, you can work from that directly.

## Scratch directory
Write intermediates to `./.pk_pe_route_scratch/<pmid>/` in the user's project /
working dir (never inside the skill folder; it is git-ignored): `identify.json`,
`design.json`, and the final `selected_pipelines.json`.

## Workflow
1. **Stage 1 — Identify** (`prompts/01_identify.md`): from the title + `abstract.md`,
   classify the paper as **PK / PE / Both / Neither** → `identify.json`.
2. **Paper-type gate** — the classification fixes the candidate set:
   - **PK** → choose only from the **PK** pipelines (`pk_*`).
   - **PE** → choose only from the **PE** pipelines (`pe_*`).
   - **Both** → choose from **both** `pk_*` and `pe_*`.
   - **Neither** → select **nothing**: write an empty `selected_pipelines.json`
     (`{"pmid": "<pmid>", "paper_type": "Neither", "selected": []}`), tell the user
     the paper is out of scope, and **stop** (do not run Stage 2).
3. **Stage 2 — Design** (`prompts/02_design.md`): within the gated candidate set,
   reconstruct the full text with tables visible (splice each `table_<n>.md` at its
   `[Table N]` marker) and select the **union** of all applicable pipelines
   (multi-label, non-exclusive) → `design.json`.
4. **Deterministic dispatch map** — never hand-write the skill names; run the
   byte-stable table:
   ```bash
   python scripts/pipeline_skill_map.py \
       --pmid <pmid> --paper-type <PK|PE|Both> <pipeline_tools from design.json> \
       > ./.pk_pe_route_scratch/<pmid>/selected_pipelines.json
   ```

## Candidate pipelines
- **PK** (`pk_*`): `pk_summary`, `pk_individual`, `pk_specimen_summary`,
  `pk_specimen_individual`, `pk_drug_summary`, `pk_drug_individual`,
  `pk_population_summary`, `pk_population_individual`.
- **PE** (`pe_*`): `pe_study_info`, `pe_study_outcome`.

## Output — `selected_pipelines.json`
```json
{ "pmid": "12345678", "paper_type": "Both",
  "selected": [
    {"pipeline": "pk_summary", "skill": "pk-summary-curation"},
    {"pipeline": "pe_study_outcome", "skill": "pe-study-outcome"} ] }
```
Each `skill` is the name of the standalone curation skill to trigger next. For a
`Neither` paper, `selected` is `[]`.

## Notes
- **Tables are visible to the design stage by design.** The legacy step saw table
  data inline in the full text; Stage 2 reconstructs that by splicing the
  `table_<n>.md` files back at their `[Table N]` markers. Do not run the design
  stage on the bare `paper_text.md` (markers only).
- The **label→skill map is deterministic** (`scripts/pipeline_skill_map.py`); only
  the identify + design judgements are model-driven.
