# Stage 2 — Select the applicable curation pipelines (multi-label)

Ports `PKPEDesignStep`. Identify **ALL** curation pipelines whose definition
matches **any** data present in the paper. Selection is **multi-label**: pipelines
are NOT mutually exclusive. Do not pick the single "best" pipeline — **maximize
coverage**.

## Input — reconstruct the full text WITH tables visible

The legacy design step saw the tables inline in the full text. Reproduce that:

1. Start from `paper_text.md`.
2. For each `[Table N]` marker, splice in the contents of the matching
   `table_<n>.md` (caption + footnotes) at the marker's position — use
   `manifest.json` for the `[Table N]` ↔ `table_<n>.md` mapping.
3. When a table's column/row structure matters to the decision (e.g. rows labeled
   by subject ID → individual-level data), also consult the corresponding
   `table_<n>.html` grid.

You also have the **title** and the **`paper_type`** from Stage 1 (`identify.json`).

## Pipeline tools

| Pipeline label | Selects when the paper contains… |
| --- | --- |
| `pk_summary` | PK summary data (means, medians, AUC, Cmax, CL, t½, …) aggregated across subjects |
| `pk_individual` | PK data whose rows correspond to subject / patient / volunteer / case IDs |
| `pk_specimen_summary` | PK summary data stratified by specimen (plasma, serum, milk, urine, tissue, …) |
| `pk_specimen_individual` | individual-level PK data with a specimen dimension |
| `pk_drug_summary` | PK summary data stratified by drug / metabolite / analyte (incl. dosing regimen) |
| `pk_drug_individual` | individual-level PK / dosing data with a drug / analyte dimension |
| `pk_population_summary` | demographic / population-level summary data (age, weight, sex, BMI, pregnancy stage, …) |
| `pk_population_individual` | individual-level demographic data |
| `pe_study_info` | PE study design / metadata |
| `pe_study_outcome` | PE outcomes |

## Operational definitions
- **Individual-level** — rows are labeled by subject ID / patient / volunteer /
  case. Includes per-subject averages and per-subject PK metrics.
- **Summary-level** — aggregates across subjects (mean, median, SD, CI).
- **Specimen-specific** — explicitly involves specimen types (plasma, serum, milk,
  urine, tissue, …).
- **Drug-specific** — distinguishes multiple drugs, metabolites, or analytes.
- **Population** — includes demographics (age, weight, sex, pregnancy stage, …).

## Selection rules (deterministic, NON-EXCLUSIVE)
1. **Domain** — PK data present → select PK pipelines; PE data present → select PE
   pipelines; both → both.
2. **Granularity** — subject-labeled rows → **individual** pipelines; aggregated
   statistics → **summary** pipelines; if both exist → BOTH.
3. **Dimension** — for each dimension present, add its pipelines: specimen →
   `pk_specimen_*`; drug/analyte → `pk_drug_*`; population/demographics →
   `pk_population_*`.
4. **Combine** — the final list is the **union** of all matched pipelines.

## Common pitfalls (do NOT)
- Choose only one pipeline.
- Prefer the "more specific" over the "general" one — select both.
- Ignore overlapping categories.

## Stability rule
- Treat **plasma-only** studies as **general PK**, not specimen-specific (do not add
  `pk_specimen_*` for plasma alone).

## Example
A table with rows = patients, reporting mean concentration, covering plasma & milk,
for parent drug + metabolite → MUST include:
`pk_summary, pk_individual, pk_specimen_summary, pk_specimen_individual,
pk_drug_summary, pk_drug_individual`.

## Output
Write `design.json` to the scratch directory:

```json
{ "pmid": "<pmid>", "pipeline_tools": ["pk_summary", "pe_study_outcome", ...],
  "reasoning": "<which dimensions/granularities you matched>" }
```

Every entry MUST be one of the 10 labels in the table above (exact spelling).

## Deterministic dispatch
Do not hand-write the procedure paths. Run the bundled map to produce the dispatch
artifact:

```bash
python pipelines/route/scripts/pipeline_skill_map.py \
    --pmid <pmid> --paper-type <PK|PE|Both> \
    <space-separated pipeline_tools from design.json> > selected_pipelines.json
```

`selected_pipelines.json` is the deliverable — it lists, for each selected
pipeline, the `skill` folder to invoke next.
