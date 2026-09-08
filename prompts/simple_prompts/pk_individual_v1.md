# pk-individual — simple single-shot prompt, v1 (control)

The original prompt, kept verbatim as the control arm of the simple-prompt
experiment. Do not "improve" this file — its value is that it is unmodified.
The revised version is `pk_individual.md`.

Output is wide (`parameter1/value1`, `parameter2/value2`, ...); the parser in
`scripts/simple_prompt/_common.py` melts it to the long gold schema, so both
arms score on identical footing.

Placeholders: `{PMID}` (unused here) and `{TABLE}`.

---

If the study is reporting PK details on individual level, report the following: 1. population: e.g. pregnant women, adult female, neonate, etc. 2. pregnancy stage: first trimester second trimester, third trimester, postpartum 3. gestational age: report the GA if report for this characteristic 4. specimen: e.g. serum, plasma, etc. 5. drug name: name of the drug 6. patient ID: ID of the patient 7. parameter1: e.g. concentration, ratio, milk:serum ration. etc. 8. drug name1: name of the drug 9. time1: time to have the specimen collected 10. time unit1: e.g. hour, days, etc. 11. value1: report the value of the characteristics 12. unit1: e.g. mg/L  if there are multiple characteristics, make additional: 13. parameter2: e.g. concentration, ratio, milk:serum ration. etc. 14. drug name2: name of the drug 15. time2: time to have the specimen collected 16. time unit2: e.g. hour, days, etc. 17. value2: report the value of the characteristics 18. unit2: e.g. mg/L and so on. Please extract as csv format

{TABLE}

---

## Known weaknesses (measured against the gold, not guessed)

These are why `pk_individual.md` exists. Listed here so the v1-vs-v2 gap is
interpretable rather than mysterious.

- **"gestational age" asks for a number.** 661 of 664 gold rows hold a category
  (`maternal`, `fetus`, `infant`, `pediatric`), not an age. This column will
  score near zero.
- **`drug name` appears twice** (items 5 and 8). Item 8 is the *analyte* - the
  substance measured, often a metabolite. Gold has `Drug name != Analyte` in
  382 of 664 rows (58%), so a model that copies the value loses most of it.
- **Pregnancy stage list is incomplete.** The gold's two most common values are
  `delivery` (202 rows) and `lactation` (104); neither is offered.
- **Wide output has a variable column count**, which makes the CSV ragged when
  different patients have different numbers of measurements.
