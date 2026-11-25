# pk-drug-individual — simple single-shot prompt

One-shot full-text-to-CSV prompt for per-individual drug dosing (the
`pk_drug_individual` pipeline). Fast path of the cascade; escalates to
`PKDrugIndWorkflow` when the check fails.

Placeholder: `{TABLE}` holds the paper's prepared full text. `{PMID}` is
substituted if present.

The 11 columns match `PKDrugIndWorkflow.go_full_text`'s output after its final
rename (`Source text` -> `Note`) and `scripts/simple_prompt/schema_specs.py`.

---

Extract the drug dosing regimen of every identified individual (patient, case, subject, mother, infant) described in the paper text below.

Output CSV with exactly these 11 columns, in this order:

Patient ID,Drug/Metabolite name,Dose amount,Dose unit,Dose frequency,Dose schedule,Dose route,Population,Pregnancy stage,Pediatric/Gestational age,Note

Write one row per individual per distinct dosing regimen. An individual whose dose changed (e.g. 8 mg then 16 mg) gets one row per regimen. Dosing that the paper reports only for a whole group with no individual identifier is not a row in this table; dosing of cases cited from other publications is not a row either.

Column rules:

- Patient ID: the identifier the paper uses for the individual: a number ("1", "2"), "Case 3", "Patient A", "Mother", "Infant", "Twin 1". Required on every row. Keep the paper's own labels; do not renumber.
- Drug/Metabolite name: the drug (or metabolite) administered. Required.
- Dose amount: the numeric amount per administration, with the unit stripped off: a value, a comma list or a range (5; 1,2,3,4; 0.01 - 0.05). Quote the cell when it contains a comma. Empty when not stated.
- Dose unit: the unit attached to the amount (mg, mg/kg, g, IU, ...).
- Dose frequency: Single, Multiple, or a number of doses (3, 4).
- Dose schedule: the timing or interval as the paper states it (once a day, twice daily, every 8 hours, at induction).
- Dose route: Oral, Intravenous (IV), Intramuscular (IM), Subcutaneous (SC), Epidural, Infusion, Sublingual, ...
- Population: Nonpregnant, Maternal (pregnant individuals), Pediatric (birth to ~17 years) or Adults (18 years or older) when one fits; otherwise the paper's wording.
- Pregnancy stage: Pre-pregnancy; Trimester 1 (up to 14 weeks); Trimester 2 (~15-28 weeks); Trimester 3 (>= 28 weeks); Fetus; Parturition, Labor or Delivery; Postpartum; Nursing, Breastfeeding or Lactation. Keep the paper's wording when none fits; empty when none is given.
- Pediatric/Gestational age: the individual's age or gestational age only when explicitly stated ("34 weeks", "6 days old"). Never inferred from a dosing or sampling time. Otherwise empty.
- Note: a short verbatim excerpt of the sentence reporting this individual's regimen (quote the cell). Empty if none can be cited.

General rules:

- Take every value from the text; never calculate, convert or invent.
- Leave a cell empty when the paper does not state it. Do not write N/A, NR or a dash.
- Quote any cell that contains a comma.
- If the paper reports no per-individual dosing, output only the header line.
- Output the CSV only: the header line and the data rows, no explanation, no markdown fence.

Paper text:

{TABLE}
