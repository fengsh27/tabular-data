# pk-individual — simple single-shot prompt

One-shot table-to-CSV prompt for individual-subject PK curation. It is the fast
path in the two-tier strategy: run this first, check the result, and escalate to
the `pk-individual-curation` skill only when the check fails.

Placeholder: `{TABLE}` (the markdown table, e.g. the `00_markdown_table.md`
produced by `pk-pe-prepare`). `{PMID}` is still substituted if present, but the
schema no longer has a PMID column - the runner knows which paper it is from
the output directory, and the per-paper CSV is column-identical to the gold.

Column vocabularies below are taken from the manual gold in
`benchmark/data/pk-individual/baseline/*_baseline_manual.csv` (664 rows over
9 papers), not invented.

---

Extract individual-subject pharmacokinetic data from the table below.

Output CSV with exactly these 12 columns, in this order:

Patient ID,Drug name,Analyte,Specimen,Population,Pregnancy stage,Pediatric/Gestational age,Parameter type,Parameter unit,Parameter value,Time value,Time unit

Write one row per single reported value. If a patient has 3 measurements, write 3 rows.

Column rules:

- Patient ID: the subject identifier from the table (a number or code), e.g. "Case 1", "Patient 3", "A". Never a citation, an author name, or a reference number. **This field is required.** If the table gives no identifier for the subject of a value, do not output that row at all - a blank Patient ID is never acceptable.
- Drug name: the drug given to the patient.
- Analyte: the substance actually measured. It is often a metabolite, so it may differ from Drug name (e.g. drug venlafaxine, analyte O-desmethylvenlafaxine).
- Specimen: plasma, serum, breast milk, umbilical cord blood, amniotic fluid, etc. This is almost never blank: if the row does not name the specimen, take it from the table caption or the column header. For a ratio between two specimens write "A/B" (e.g. "umbilical cord/maternal plasma"). When the value is not a measurement in a matrix at all - a dose ratio, for instance - leave it blank, but still write its comma.
- Population: who the subject is - maternal, pediatric, or maternal/pediatric. Use the compound "maternal/pediatric" only when one row genuinely covers both subjects, such as a cord-to-maternal ratio; otherwise choose one. Blank if the table does not say.
- Pregnancy stage: when the sample was taken - delivery, lactation, pregnancy, postpartum, 1st trimester, 2nd trimester, 3rd trimester. Blank if not stated.
- Pediatric/Gestational age: this is NOT a number. It says which body the value belongs to: maternal, fetus, infant, pediatric, or maternal/pediatric. Leave it BLANK unless the table itself distinguishes the subjects - for example separate maternal and cord-blood columns, or an infant row. Most rows leave this blank. Never join two values with a comma. Write an actual age (e.g. "6 months", "38w0d") only if the table gives one for that individual.
- Parameter type: what was measured. Use "concentration" for a plain measured concentration. Use Cmax, tmax, or AUC when the table names them. For a ratio, name the ratio (e.g. "umbilical maternal ratio", "M/P AUC", "concentration-dose ratio"). Do not put the specimen or the subject in this field - they have their own columns.
- Parameter unit: ng/ml, ug/l, ug/ml, %, h, etc.
- Parameter value: the number as printed. If the cell is "value (range)", put the main value here.
- Time value / Time unit: only when the table gives an explicit sampling time (e.g. 12, h). Usually blank - leave both blank rather than guessing.

Rules:

- Individual-subject data only. Every row must trace to one person identified in the table. Skip means, medians, and group summaries.
- Skip rows that report cases published in OTHER papers. The tell is a per-row
  citation, author name, or bracketed reference number (e.g.
  "Klenske et al. 2019 [12]"), often sitting in a column whose header is empty
  and therefore rendered as `Unnamed_0`. A table of such rows is a literature
  review, not this paper's own subjects - take nothing from it, however much
  the caption ("Cases of ...") and the units columns look like patient data.
- Use only the table below. Do not add data from anywhere else.
- If no row in the table qualifies, output the header row and nothing else.
- Leave a cell empty when the table does not report it. Never write N/A, NA, or "not reported".
- Output the header row, then the data rows.
- **Every row must have exactly 11 commas.** An empty field is still a field:
  write its comma. A dose-ratio row for patient 1 with no specimen and no
  timing is
  `1,venlafaxine,venlafaxine,,maternal/pediatric,,,infant/maternal dose ratio,%,5.9,,`
  - the empty Specimen, Pregnancy stage, Pediatric/Gestational age, Time value
  and Time unit each still get their comma. Leave one out and every later
  column shifts by one, which makes the whole row wrong. Count the commas
  before you write the row.
- Output only the CSV. No commentary, no code fences.

TABLE:

{TABLE}

---

## Known gaps against the gold

- `Parameter type` in the gold is not a clean controlled vocabulary.
  `concentration` covers 51% of rows, but the tail is descriptive
  (`penetration ratio of cord blood/maternal serum`,
  `concentration of 12 weeks' postpartum`, and `infant plasma concentration`,
  which contradicts the "no subject in this field" rule). Expect loss on the tail.
- `Population` and `Pediatric/Gestational age` overlap in the gold - both carry
  `maternal` / `pediatric`. The distinction between them is not well defined,
  so both columns will be noisy for any model.
- `Pregnancy stage` in the gold is inconsistently cased and contains a typo
  (`3rd trimester` vs `third trimester`, `Postpartum` vs `postpartum`,
  `2nd trimster`). Score this column case-insensitively.
