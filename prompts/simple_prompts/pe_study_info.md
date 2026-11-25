# pe-study-info — simple single-shot prompt

One-shot full-text-to-CSV prompt for pharmacoepidemiology study-level
metadata (the `pe_study_info` pipeline). Fast path of the cascade; escalates
to `PEStudyInfoWorkflow` when the check fails.

Placeholder: `{TABLE}` holds the paper's prepared full text. `{PMID}` is
substituted if present. The output is a single row.

The 10 columns are `benchmark/configs.py`'s PE_STUDY_INFO_COLUMNS_TYPE (the
same names `PEStudyInfoWorkflow.go_full_text` emits, in the benchmark's order)
and `scripts/simple_prompt/schema_specs.py`.

---

Summarise the design of the study described in the paper text below as ONE row.

Output CSV with exactly these 10 columns, in this order:

Study type,Population,Study design,Pregnancy stage,Drug name,Data source,Inclusion criteria,Exclusion criteria,Outcomes,Subject N

Write exactly one data row describing the study. Never more than one row.

Column rules:

- Study type: the area of interest of the article, one of: Pharmacoepidemiology, Clinical Trials, Pharmacokinetics, Pharmacodynamics, Pharmacogenetics. Required.
- Population: the subjects studied: Nonpregnant, Maternal (pregnant individuals), Pediatric (birth to ~17 years) or Adults (18 years or older) when one fits; several separated by "; " when the study covers more than one (e.g. "Maternal; Pediatric"); otherwise the paper's wording.
- Study design: the design as the paper describes it, e.g. Prospective cohort study, Retrospective cohort study, Randomized controlled trial (RCT), Double-blind randomized trial, Case-control study, Cross-sectional study, Systematic review and meta-analysis, Open-label study, Nested case-control study, Pilot study, Chart review, Observational study. When several apply ("prospective, randomized, double-blind") write them as one string in the paper's order.
- Pregnancy stage: the stage the study covers: Trimester 1 (up to 14 weeks); Trimester 2 (~15-28 weeks); Trimester 3 (>= 28 weeks); Fetus or Fetal stage; Parturition, Labor or Delivery; Postpartum (~6-8 weeks after birth); Nursing, Breastfeeding or Lactation. Several separated by "; ". Keep the paper's wording when none fits; empty when none is given.
- Drug name: every drug of interest in relation to the study outcomes, separated by "; " when there are several.
- Data source: where the data came from or where the study was conducted: hospitals, clinics, registries, databases, cohorts, geographic locations (e.g. "Ohio State University Medical Center", "Medicaid claims database, Tennessee").
- Inclusion criteria: the criteria participants had to meet, in the paper's own wording (quote the cell).
- Exclusion criteria: the criteria that disqualified participants, in the paper's own wording (quote the cell).
- Outcomes: the outcome measure(s) the investigators considered most important, in the paper's own wording (quote the cell).
- Subject N: the total number of subjects studied, as a plain number.

General rules:

- Take every value from the text; never invent. Leave a cell empty when the paper does not state it. Do not write N/A, NR or a dash.
- Quote any cell that contains a comma (the criteria and outcome cells almost always will).
- Output the CSV only: the header line and the one data row, no explanation, no markdown fence.

Paper text:

{TABLE}
