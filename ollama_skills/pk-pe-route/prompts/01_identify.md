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
- **Abstract** — `abstract.md`. It is **not always present**: `manifest.json`
  reports `"has_abstract": false` when the paper had none (or none could be
  extracted). In that case do not stop — fall back to the title plus the opening
  paragraphs of `paper_text.md`, and note in `reasoning` that you classified
  without an abstract.
- **Table digest** — run this once, from the skill folder, on **every** paper:

  ```bash
  python scripts/table_digest.py <prepared-paper-dir>
  ```

  It prints, per table, the caption + footnotes and the table's header plus its
  first couple of rows (raise `--max-rows` if you need more, `0` for the whole
  table). Read it on
  every paper, not just borderline ones: a paper's PK content frequently lives
  only in its tables, and the tell may be
  - a **caption** — "Pharmacokinetic parameters of…",
  - a **column header** — `Cmax | AUC | t½ | CL`, **or**
  - a **row label** — "Mean serum fentanyl concentration (nmol/L)".

  The digest covers all three. In this corpus parameter names are often row
  labels rather than column headers, so do not look only at the header row.

  The digest truncates long tables and says so. If a table looks relevant but
  was cut off, read that table's `table_<n>.md` — it holds the caption,
  footnotes, and the **full** table as Markdown. You should not need
  `table_<n>.html` at this stage; it is the same table in a much larger form.
  If `manifest.json` reports no tables, the digest says so — skip this input.
- **Full results text** — read `paper_text.md` **only when you are about to answer
  `Neither`** (the gate below), to confirm no analyte measurement was missed. Do
  not pull the whole prose in for routine PK/PE/Both calls; it slows the gate and
  adds noise without improving the judgement.

Read these from the prepared-paper directory; do not rely on the conversation.

## Definitions (read carefully — broader than "drugs only")
- **Pharmacokinetics (PK)** — the **time-course or disposition** of **any
  analyte** in the body. The paper is PK if it reports **any one** of:
  - a **derived PK parameter** — clearance, half-life, AUC, Cmax, Tmax, volume of
    distribution, bioavailability, turnover / metabolic flux; **or**
  - concentrations sampled **over time**, or tied to a **dose** or to a **time
    after dose**; **or**
  - concentrations compared across compartments **to quantify transfer or
    distribution** — cord vs. maternal blood, milk vs. plasma, tissue vs. serum.
    The transfer has to be the point: the paper is asking how much of the analyte
    crosses between compartments, or where in the body it ends up.

    **Not a compartment comparison:** two anatomical sub-regions of one organ
    (e.g. two brain regions on MRS), or several tissues each sampled once to
    describe status. Measuring an analyte in more than one place is not the same
    as studying its movement between those places.

  The analyte may be:
  - an **administered drug or xenobiotic**, **or**
  - an **endogenous compound** — a hormone (e.g. LH, insulin), metabolite, or
    other native substance whose concentration or kinetics are measured, **or**
  - a **dietary / supplemental agent** — a nutrient, vitamin, mineral, or
    supplement given to subjects (treat these as administered agents), **or**
  - a **stable-isotope tracer** used to quantify turnover, synthesis, or
    metabolic rates.

  Measured in plasma, serum, urine, tissue, or any specimen, in **humans or
  animals** (preclinical/animal PK counts).

  **Not PK on its own:** a **concentration measured only to describe or classify
  subjects** — a nutrient status, a baseline biomarker, a routine lab value.
  Measuring an analyte is not the same as studying its kinetics.

  **What the measurements are for decides it.** Repeating a measurement — at
  several visits, in several tissues, in several people — does not by itself make
  a paper PK. Ask what the measurements exist to do. If they characterize the
  analyte's behaviour in the body (how fast it falls, how much reaches the fetus,
  what its AUC is), that is PK. If they exist **only** to sort subjects into
  groups whose outcomes are then compared, that is the exposure side of an
  exposure→outcome study — step 3 of the Task governs, and the answer is `PE`.
  A paper can do **both** — report real kinetics *and* compare outcomes across
  groups; that is `Both`, and step 4 governs. The test rules out PK only when
  classifying subjects is the sole reason the analyte was measured.
- **Pharmacoepidemiology (PE)** — the use and effects of drugs, supplements, or
  other modifiable agents in **populations**: observational / real-world data
  (cohort, case-control, claims, EHR) or trials that relate an **exposure** to
  clinical outcomes, safety, utilization, adherence, effectiveness, or
  risk–benefit. Supplement/nutrient exposures studied for their effect on
  outcomes count.

## Task
Classify the paper as exactly one of `PK`, `PE`, `Both`, or `Neither`.

Work in this order:
1. **Look for PK signal** (title + abstract + table digest): does the paper
   report an analyte's **kinetics** — a PK parameter, concentrations over time or
   against dose, or a compartment-to-compartment comparison? If yes → `PK`.
2. **Look for PE signal**: is an administered/modifiable exposure related to
   outcomes in a population? If yes → `PE`.
3. **A measured level alone does not settle it.** If an analyte's level is
   measured only to **describe or classify subjects**, and the paper's question
   is exposure→outcome, that is `PE`, **not** `PK` (see the vitamin-D row below).
4. **If both 1 and 2 hold, answer `Both`** — do not pick whichever seems
   stronger. `Both` is the right answer for a paper that reports kinetics *and*
   relates an exposure to outcomes.
5. **Only if 1 and 2 are both clearly absent**, consider `Neither` — and first
   pass the gate below.

## Neither gate (must clear ALL before answering `Neither`)
You may answer `Neither` only if **every** statement below is true. If any is
false (or you are unsure), pick `PK`, `PE`, or `Both` instead.

The first item below is deliberately **broader** than the PK definition above: a
bare concentration is not enough to call a paper `PK`, but it *is* enough to stop
you dropping the paper entirely. This gate decides whether the paper is discarded,
not which label it gets.
- [ ] No table or section reports a **concentration, level, or PK parameter**
      (clearance, half-life, AUC, Cmax, Tmax, Vd, turnover/flux) for any analyte.
- [ ] No **drug, supplement, hormone, nutrient, or tracer** is administered,
      dosed, or measured.
- [ ] No analysis relates an **exposure to an outcome** in a population.

All three are deliberately broad, so **clearing this gate is rare** — that is the
design, not a failure on your part. A paper that does clear it looks like a
surgical-technique or device study, an imaging study reporting only anatomy
(volumes, lesion counts), or a methods / questionnaire-validation paper — none of
which administer or measure an analyte.

**If the gate blocks `Neither` but neither step 1 nor step 2 held** — e.g. a
descriptive paper that reports analyte levels but studies no kinetics and no
exposure→outcome — answer **`PK`**. The measured levels are the only curatable
signal, and the `pk_*` pipelines are the ones that can use them.

## Edge cases (decided for this corpus)
| Paper | Label | Why |
|-------|-------|-----|
| Endogenous hormone kinetics in animals (e.g. **LH half-life/clearance in monkeys**) | **PK** | half-life/clearance of an analyte = PK; endogenous & animal both count |
| **Isotope-tracer** turnover (e.g. [13C]methionine flux in neonates) | **PK** | tracer kinetics quantify metabolic rates = PK |
| **Supplement RCT** that samples the analyte **over time** or derives a PK parameter (e.g. zinc supplementation with a concentration–time profile) | **PK** | nutrient = administered agent, and the kinetics are reported |
| **Supplement RCT** where the biomarker is just the **endpoint** — one post-treatment level per subject (e.g. zinc supplementation, final RBC metallothionein) | **PE** | the level classifies the result; no kinetics are reported, so it is an exposure→outcome trial |
| **Nutrient-status → outcome** observational study (e.g. vitamin-D status vs. gestational diabetes) | **PE** | the level classifies the exposure; the study question is exposure→outcome |
| Drug concentrations **and** clinical outcomes compared across a real-world cohort (e.g. vancomycin Cavg before/after a dosing protocol, with clinical parameters) | **Both** | dose-tied concentrations = PK; retrospective exposure→outcome comparison = PE |
| **Surgical-technique or device** study predicting outcome, no analyte administered or measured (e.g. a comparison of two catheter placements) | **Neither** | clears every gate item |

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
