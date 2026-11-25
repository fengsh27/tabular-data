# pk-specimen-individual — simple single-shot prompt

One-shot full-text-to-CSV prompt for per-individual specimen sampling (the
`pk_specimen_individual` pipeline). Fast path of the cascade; escalates to
`PKSpecIndWorkflow` when the check fails.

Placeholder: `{TABLE}` holds the paper's prepared full text. `{PMID}` is
substituted if present.

The 9 columns match `PKSpecIndWorkflow.go_full_text`'s output after its final
rename (`Source text` -> `Note`) and `scripts/simple_prompt/schema_specs.py`.

---

Extract the specimen sampling of every identified individual (patient, case, subject, mother, infant) described in the paper text below.

Output CSV with exactly these 9 columns, in this order:

Patient ID,Specimen,Sample N,Population,Pregnancy stage,Pediatric/Gestational age,Sample time,Time unit,Note

Write one row per individual per specimen type per separately described sampling scheme. Sampling that the paper reports only for a whole group with no individual identifier is not a row in this table.

Column rules:

- Patient ID: the identifier the paper uses for the individual: a number ("1", "2"), "Case 3", "Patient A", "Mother", "Infant". Required on every row. Keep the paper's own labels.
- Specimen: the biological matrix collected: blood, plasma, serum, urine, saliva, breast milk, umbilical cord blood, amniotic fluid, meconium, placenta, CSF, ... Required.
- Sample N: the number of samples of that specimen taken from this individual, as a plain number, only when explicitly stated. Never derived by calculation.
- Population: Nonpregnant, Maternal (pregnant individuals), Pediatric (birth to ~17 years) or Adults (18 years or older) when one fits; otherwise the paper's wording.
- Pregnancy stage: Pre-pregnancy; Trimester 1 (up to 14 weeks); Trimester 2 (~15-28 weeks); Trimester 3 (>= 28 weeks); Fetus; Parturition, Labor or Delivery; Postpartum; Nursing, Breastfeeding or Lactation. Keep the paper's wording when none fits; empty when none is given.
- Pediatric/Gestational age: the individual's age or gestational age only when explicitly stated. Never inferred from a sampling time. Otherwise empty.
- Sample time: when the specimen was taken, as numbers only: a value, a comma list ("0, 2, 4") or a range ("0-2"). Quote the cell when it contains a comma. Descriptive timing with no number goes in Note and the cell stays empty.
- Time unit: the unit of Sample time (Second, Minute, Hour, Day, Week). Empty when Sample time is empty.
- Note: a short verbatim excerpt of the sentence describing this sampling (quote the cell). Empty if none can be cited.

General rules:

- Take every value from the text; never calculate, convert or invent.
- Leave a cell empty when the paper does not state it. Do not write N/A, NR or a dash.
- Quote any cell that contains a comma.
- If the paper reports no per-individual sampling, output only the header line.
- Output the CSV only: the header line and the data rows, no explanation, no markdown fence.

Paper text:

{TABLE}
