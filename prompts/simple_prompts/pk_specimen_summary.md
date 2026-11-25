# pk-specimen-summary — simple single-shot prompt

One-shot full-text-to-CSV prompt for study-level specimen sampling (the
`pk_specimen_summary` pipeline). Fast path of the cascade; escalates to
`PKSpecSumWorkflow` when the check fails.

Placeholder: `{TABLE}` holds the paper's prepared full text. `{PMID}` is
substituted if present.

The 9 columns match `PKSpecSumWorkflow.go_full_text`'s output after its final
rename (`Population N` -> `Subject N`, `Source text` -> `Note`) and
`scripts/simple_prompt/schema_specs.py`.

---

Extract how biological specimens were collected for the subjects of the study described in the paper text below.

Output CSV with exactly these 9 columns, in this order:

Specimen,Sample N,Population,Pregnancy stage,Pediatric/Gestational age,Subject N,Sample time,Time unit,Note

Write one row per unique combination of specimen type, population group, sampling time scheme and sample count that the paper states for its own study. The same specimen collected from two groups, or at two separately described times, is two rows.

Column rules:

- Specimen: the biological matrix collected: blood, plasma, serum, whole blood, urine, saliva, breast milk, umbilical cord blood, amniotic fluid, meconium, placenta, CSF, ... Required on every row.
- Sample N: the number of samples of that specimen analysed, as a plain number, only when the paper states it explicitly. Never derive it by calculation. When the text gives per-group or per-time counts AND their total, write the individual counts (one row each) and leave the total out.
- Population: Nonpregnant, Maternal (pregnant individuals), Pediatric (birth to ~17 years) or Adults (18 years or older) when one fits; otherwise the paper's wording (e.g. "neonates", "healthy volunteers").
- Pregnancy stage: Pre-pregnancy; Trimester 1 (up to 14 weeks); Trimester 2 (~15-28 weeks); Trimester 3 (>= 28 weeks); Fetus; Parturition, Labor or Delivery; Postpartum (~6-8 weeks after birth); Nursing, Breastfeeding or Lactation. Keep the paper's wording when none fits; empty when none is given.
- Pediatric/Gestational age: the age or gestational age of the sampled group only when explicitly stated ("32-36 weeks", "<48 hours of life"). Never inferred from a sampling time. Otherwise empty.
- Subject N: the number of individuals in that population group the specimen was collected from, as a plain number. Empty when not stated.
- Sample time: when the specimen was taken, as numbers only: a single value ("0", "24"), a comma list ("0, 2, 4") or a range ("0-2", "0-2, 2-4"). Quote the cell when it contains a comma. Descriptive timing with no number ("at delivery", "before the morning dose") goes in Note and the cell stays empty.
- Time unit: the unit of Sample time (Second, Minute, Hour, Day, Week). Empty when Sample time is empty.
- Note: a short verbatim excerpt of the sentence describing this sampling (quote the cell). Empty if none can be cited.

General rules:

- Take every value from the text; never calculate, convert or invent.
- Leave a cell empty when the paper does not state it. Do not write N/A, NR or a dash.
- Quote any cell that contains a comma.
- If the paper describes no specimen collection of its own, output only the header line.
- Output the CSV only: the header line and the data rows, no explanation, no markdown fence.

Paper text:

{TABLE}
