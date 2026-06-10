# Stage 1 — Identify the paper type (PK / PE / Both / Neither)

Ports `PKPEIdentificationStep`. Decide whether the paper is pharmacokinetics-,
pharmacoepidemiology-related, both, or neither — from the **title and abstract
only**. This is a cheap gate: if the answer is `Neither`, routing stops and no
curation pipeline runs.

## Inputs
- **Title** — the H1 on the first line of `paper_text.md` (or the `title` field in
  `manifest.json`).
- **Abstract** — `abstract.md`.

Read both from the prepared-paper directory; do not rely on the conversation.

## Definitions
- **Pharmacokinetics (PK)** — how a drug is absorbed, distributed, metabolized, and
  excreted. PK studies report parameters such as clearance, half-life, AUC, Cmax,
  volume of distribution, bioavailability; designs measure drug concentrations in
  plasma/tissue over time.
- **Pharmacoepidemiology (PE)** — the use and effects of drugs in large
  populations. PE studies use observational / real-world data (claims, EHR) and
  focus on drug safety, utilization, adherence, effectiveness, risk–benefit, and
  post-marketing surveillance.

## Task
Classify the paper as exactly one of:
- `PK` — pharmacokinetics-related
- `PE` — pharmacoepidemiology-related
- `Both` — related to both PK and PE
- `Neither` — related to neither

Think step by step about which definitions the title + abstract match, then commit
to a single label.

## Output
Write `identify.json` to the scratch directory:

```json
{ "pmid": "<pmid>", "paper_type": "PK | PE | Both | Neither",
  "reasoning": "<1-2 sentences>" }
```

**If `paper_type` is `Neither`:** stop here. Write an empty selection
(`{"pmid": "<pmid>", "paper_type": "Neither", "selected": []}`) to
`selected_pipelines.json`, tell the user the paper is neither PK nor PE, and do
**not** run Stage 2.

Otherwise continue to Stage 2 (`02_design.md`).
