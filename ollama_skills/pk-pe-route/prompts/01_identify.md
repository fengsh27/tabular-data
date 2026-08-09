# Stage 1 — Identify the paper type (PK / PE / Both / Neither)

Ports `PKPEIdentificationStep`. Decide whether the paper is pharmacokinetics-,
pharmacoepidemiology-related, both, or neither. This is a gate: if the answer is
`Neither`, routing stops and **no curation pipeline runs**, so the paper is
dropped entirely — treat a `Neither` verdict as a high-bar decision (see the
Neither gate below).

## Context — what this corpus is
These papers come from a **pre-curated PK/PE set**. The large majority *are*
PK and/or PE; a paper that is genuinely neither is the exception. A wrong
`Neither` silently discards a curatable paper (zero recall) and is the most
costly error here, so **when in doubt, prefer `PK`/`PE`/`Both` over `Neither`.**

## Inputs
- **Title** — the H1 on the first line of `paper_text.md` (or the `title` field in
  `manifest.json`).
- **Abstract** — `abstract.md`.
- **Table digest** — run this once, from the skill folder, on **every** paper:

  ```bash
  python scripts/table_digest.py <prepared-paper-dir>
  ```

  It prints, per table, the caption + footnotes and the first 3 rows. Read it on
  every paper, not just borderline ones: a paper's PK content frequently lives
  only in its tables, and the tell may be
  - a **caption** — "Pharmacokinetic parameters of…",
  - a **column header** — `Cmax | AUC | t½ | CL`, **or**
  - a **row label** — "Mean serum fentanyl concentration (nmol/L)".

  The digest covers all three. In this corpus parameter names are often row
  labels rather than column headers, so do not look only at the header row.

  Do **not** try to get this from the prepared files by hand: `table_<n>.md`
  holds only the caption and footnotes (**no table rows at all**), and
  `table_<n>.html` holds the entire table and runs 7–18× larger than the digest.
  Open `table_<n>.html` directly only when the digest is genuinely not enough.
  If `manifest.json` reports no tables, the digest says so — skip this input.
- **Full results text** — read `paper_text.md` **only when you are about to answer
  `Neither`** (the gate below), to confirm no analyte measurement was missed. Do
  not pull the whole prose in for routine PK/PE/Both calls; it slows the gate and
  adds noise without improving the judgement.

Read these from the prepared-paper directory; do not rely on the conversation.

## Definitions (read carefully — broader than "drugs only")
- **Pharmacokinetics (PK)** — the time-course and disposition of **any analyte**
  in the body: its concentration, clearance, half-life, AUC, Cmax, Tmax, volume
  of distribution, bioavailability, or turnover / metabolic flux. The analyte may
  be:
  - an **administered drug or xenobiotic**, **or**
  - an **endogenous compound** — a hormone (e.g. LH, insulin), metabolite, or
    other native substance whose concentration or kinetics are measured, **or**
  - a **dietary / supplemental agent** — a nutrient, vitamin, mineral, or
    supplement given to subjects (treat these as administered agents), **or**
  - a **stable-isotope tracer** used to quantify turnover, synthesis, or
    metabolic rates.

  Measured in plasma, serum, urine, tissue, or any specimen, in **humans or
  animals** (preclinical/animal PK counts).
- **Pharmacoepidemiology (PE)** — the use and effects of drugs, supplements, or
  other modifiable agents in **populations**: observational / real-world data
  (cohort, case-control, claims, EHR) or trials that relate an **exposure** to
  clinical outcomes, safety, utilization, adherence, effectiveness, or
  risk–benefit. Supplement/nutrient exposures studied for their effect on
  outcomes count.

## Task
Classify the paper as exactly one of `PK`, `PE`, `Both`, or `Neither`.

Work in this order:
1. **Look for PK signal** (title + abstract + table digest): is *any* analyte's
   concentration, kinetics, or turnover measured/reported? If yes → the paper is
   `PK` (or `Both`).
2. **Look for PE signal**: is an administered/modifiable exposure related to
   outcomes in a population? If yes → `PE` (or `Both`).
3. **Only if both are clearly absent**, consider `Neither` — and first pass the
   gate below.

## Neither gate (must clear ALL before answering `Neither`)
You may answer `Neither` only if **every** statement below is true. If any is
false (or you are unsure), pick `PK`, `PE`, or `Both` instead.
- [ ] No table or section reports a **concentration, level, or PK parameter**
      (clearance, half-life, AUC, Cmax, Tmax, Vd, turnover/flux) for any analyte.
- [ ] No **drug, supplement, hormone, nutrient, or tracer** is administered,
      dosed, or measured.
- [ ] No analysis relates an **exposure to an outcome** in a population.
- [ ] The paper is plausibly outside both definitions even after reading its
      tables (e.g. a pure imaging, surgical-technique, genetics, or methods paper
      with no analyte measurement).

## Edge cases (decided for this corpus)
| Paper | Label | Why |
|-------|-------|-----|
| Endogenous hormone kinetics in animals (e.g. **LH half-life/clearance in monkeys**) | **PK** | half-life/clearance of an analyte = PK; endogenous & animal both count |
| **Isotope-tracer** turnover (e.g. [13C]methionine flux in neonates) | **PK** | tracer kinetics quantify metabolic rates = PK |
| **Supplement RCT** measuring a biomarker concentration (e.g. zinc supplementation, RBC metallothionein) | **PK** (or **Both** if it also tests exposure→outcome) | nutrient = administered agent; measured concentration = PK |
| **Nutrient-status → outcome** observational study (e.g. vitamin-D status vs. gestational diabetes) | **PE** | modifiable exposure related to outcomes in a population |
| Pure MR-spectroscopy / imaging predicting outcome, **no analyte concentration** | **Neither** | clears every gate item |

## Output
Write `identify.json` to the scratch directory:

```json
{ "pmid": "<pmid>", "paper_type": "PK | PE | Both | Neither",
  "reasoning": "<1-2 sentences; if Neither, state which gate items were checked>" }
```

**If `paper_type` is `Neither`:** stop here. Write an empty selection
(`{"pmid": "<pmid>", "paper_type": "Neither", "selected": []}`) to
`selected_pipelines.json`, tell the user the paper is neither PK nor PE, and do
**not** run Stage 2.

Otherwise continue to Stage 2 (`02_design.md`).
