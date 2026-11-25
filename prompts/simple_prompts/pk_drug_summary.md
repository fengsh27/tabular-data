# pk-drug-summary — simple single-shot prompt

One-shot full-text-to-CSV prompt for study-level drug dosing (the
`pk_drug_summary` pipeline). Fast path of the cascade: run this first, check
the result, escalate to `PKDrugSumWorkflow` only when the check fails.

Placeholder: `{TABLE}` holds the paper's prepared full text (title, sections,
references stripped), the same text `FullTextCurationTool` hands the
production workflow. `{PMID}` is substituted if present.

The 11 columns match `PKDrugSumWorkflow.go_full_text`'s output after its final
rename (`Population N` -> `Subject N`, `Source text` -> `Note`), i.e.
`benchmark/data/pk-pe/<version>/{pmid}_pk_drug_summary_<model>.csv`, and the
column roles in `scripts/simple_prompt/schema_specs.py`.

---

Extract every drug dosing regimen given to the subjects of the study described in the paper text below.

Output CSV with exactly these 11 columns, in this order:

Drug/Metabolite name,Dose amount,Dose unit,Dose frequency,Dose schedule,Dose route,Population,Pregnancy stage,Pediatric/Gestational age,Subject N,Note

Write one row per unique combination of drug, dose amount, dose frequency, population group and subject count that the paper states for its own study subjects. Two groups given the same regimen are two rows (their Population or Subject N differ). A regimen the paper only cites from another study is not a row.

Column rules:

- Drug/Metabolite name: the drug (or metabolite) that was administered. Required on every row.
- Dose amount: the numeric amount per administration, with the unit stripped off: a value, a comma list or a range (5; 1,2,3,4; 0.01 - 0.05). Quote the cell when it contains a comma. Leave empty when the paper gives no amount.
- Dose unit: the unit that was attached to the amount (mg, mg/kg, g, IU, ...).
- Dose frequency: how many times the drug was taken: Single, Multiple, or a number of doses (3, 4).
- Dose schedule: the timing or interval of administration as the paper states it (once a day, twice daily, every 8 hours, at induction, continuous infusion over 24 h).
- Dose route: Oral, Intravenous (IV), Intramuscular (IM), Subcutaneous (SC), Epidural, Infusion, Sublingual, Topical, Inhalation, ...
- Population: the clinical group that received the regimen. Use Nonpregnant, Maternal (pregnant individuals), Pediatric (birth to ~17 years) or Adults (18 years or older) when one of them fits; otherwise keep the paper's wording (e.g. "neonates", "healthy volunteers").
- Pregnancy stage: Pre-pregnancy; Trimester 1 (up to 14 weeks); Trimester 2 (~15-28 weeks); Trimester 3 (>= 28 weeks); Fetus; Parturition, Labor or Delivery; Postpartum (~6-8 weeks after birth); Nursing, Breastfeeding or Lactation. Keep the paper's wording when none fits; empty when the paper gives none.
- Pediatric/Gestational age: the child's age or the gestational age only when it is explicitly stated for that group (e.g. "32-36 weeks", "2-12 months"). Never infer it from a dosing or sampling time. Otherwise empty.
- Subject N: the number of individuals in that population group who received the regimen, as a plain number. Empty when not stated.
- Note: a short verbatim excerpt of the sentence that reports this regimen (quote the cell; it will contain commas). Empty if no sentence can be cited.

General rules:

- Take every value from the text; never calculate, convert or invent.
- Leave a cell empty when the paper does not state it. Do not write N/A, NR or a dash.
- Quote any cell that contains a comma.
- If the paper describes no dosing at all for its own subjects, output only the header line.
- Output the CSV only: the header line and the data rows, no explanation, no markdown fence.

Paper text:

{TABLE}
