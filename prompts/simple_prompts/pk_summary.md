# pk-summary — simple single-shot prompt

One-shot table-to-CSV prompt for cohort/group-level summary PK curation. Fast
path in the two-tier strategy: run this first, check the result, and escalate
to the `pk-summary-curation` skill only when the check fails.

Placeholder: `{TABLE}` (the markdown table, e.g. the `00_markdown_table.md`
produced by `pk-pe-prepare`). `{PMID}` is still substituted if present, but the
schema has no PMID column - the runner knows which paper it is from the output
directory.

Column names and the 16-column set below match the existing gold in
`benchmark/data/pk-summary/baseline/<pmid>_baseline.csv` (and the scorer's
canonical column list in `benchmark/configs.py`), not the newer 19-column
`pk-summary-curation` skill schema - `Pediatric/Gestational age`, `Time value`
and `Time unit` are intentionally not columns here.

---

Extract cohort/group-level summary pharmacokinetic data from the table below.

Output CSV with exactly these 16 columns, in this order:

Drug name,Analyte,Specimen,Population,Pregnancy stage,Subject N,Parameter type,Value,Unit,Summary statistics,Variation type,Variation value,Interval type,Lower limit,High limit,P value

Write one row per single reported statistic. If a cohort has both a mean and a
median reported for the same parameter, write 2 rows.

Column rules:

- Drug name: the drug administered in the study. Many summary tables never
  repeat the drug name in the table body at all (a single-drug paper states
  it once, in the title or caption, and every row of every table is
  implicitly about that drug) - when the table itself does not name it, take
  it from the paper title or the table caption instead. Leave this blank only
  if neither the table, its caption, nor the paper title names a drug.
- Analyte: the substance actually measured. Often a metabolite, so it may
  differ from Drug name (e.g. drug venlafaxine, analyte
  O-desmethylvenlafaxine). Same fallback as Drug name: when the table does not
  name the analyte and gives no reason to think it differs from the drug, use
  the drug name found via that fallback.
- Specimen: plasma, serum, blood, urine, breast milk, umbilical cord blood,
  CSF, etc. If the row does not name it, take it from the table caption or the
  column header.
- Population: the clinical population type - e.g. "maternal", "pediatric",
  "healthy adults", "child", "neonates". Use only this kind of clinical /
  demographic label. A table often ALSO splits its statistics across several
  study-defined comparison groups within that same clinical population - a
  before/after cohort, a treatment vs. control arm, a with/without-condition
  split such as "patients with ARC" vs. "patients without ARC", an age
  stratum such as "3 Month to < 3 Years", or a group-header row like "Overall"
  that introduces one such block. That comparison-group distinction has no
  column of its own in this schema: do not put it in Population, do not put
  it in Pregnancy stage, and do not invent a new column for it - write the
  SAME clinical population value on every row that shares it, and let Subject
  N (which does differ row to row) and the row's own Parameter type / Value
  keep the groups apart. Blank only when the table gives no population
  information at all.
- Pregnancy stage: when the sample was taken - delivery, lactation, pregnancy,
  postpartum, 1st trimester, 2nd trimester, 3rd trimester. Blank if not
  stated or not an obstetric study.
- Subject N: the integer number of subjects the row's statistic was computed
  over. Blank if the table does not give a count for this specific row (a
  table-wide "N=20" in the caption applies to every row that shares its
  cohort).
- Parameter type: what was measured - Cmax, tmax, AUC0-t, AUC0-∞, t1/2, CL,
  Vd, Ka, etc. Use the table's own abbreviation when it gives one. Do not put
  the specimen, population, or statistic type in this field - they have their
  own columns.
- Value: the central numeric value as printed (the mean, median, geometric
  mean, etc. - whichever this row's Summary statistics says).
- Unit: ng/ml, ug/l, h, L/h, ml/min/kg, %, etc.
- Summary statistics: what Value represents - mean, median, geometric mean.
  This field is required whenever Value is filled: never leave a row with a
  Value and no Summary statistics.
- Variation type: the variability measure reported alongside Value - SD, CV%,
  SEM, range. Blank if none is reported for this row.
- Variation value: the single number for Variation type (e.g. the SD itself).
  Blank if Variation type is blank.
- Interval type: 95% CI, range, IQR, min-max. Blank if the table gives no
  interval for this row.
- Lower limit / High limit: the two ends of Interval type, as printed. Both
  blank if Interval type is blank.
- P value: only when the table reports a p-value for this specific row's
  comparison. Usually blank.

Rules:

- Cohort/group-level summary data only. Every row must be a statistic
  computed across a group (mean, median, SD, range, CI, ...) - never a single
  named subject's own measurement. Skip any table, or any column within a
  table, that reports one value per identified individual patient - that is
  pk-individual data, not this schema.
- Skip rows that report cases published in OTHER papers. The tell is a
  per-row citation, author name, or bracketed reference number (e.g.
  "Klenske et al. 2019 [12]"), often sitting in a column whose header is
  empty and therefore rendered as `Unnamed_0`. A table of such rows is a
  literature review, not this paper's own data - take nothing from it.
- Skip tables that report only demographics or baseline characteristics with
  no pharmacokinetic parameter (no Cmax/AUC/CL/t1/2/... and no drug
  concentration). Age, weight, and dosing-regimen tables are not PK summary
  data.
- A mean/median and its OWN variation or interval belong on the SAME row, not
  split across separate rows. Some tables transpose the layout so statistic
  names are row labels and parameters are columns (rows "N" / "Range" /
  "Mean ± s.d." / "Median", one column per parameter). Read such a table
  column by column: for one parameter, its N row gives Subject N, its
  "Mean ± s.d." row gives Value + Summary statistics "mean" + Variation type
  "SD" + Variation value, and its "Range" row folds into that SAME mean row as
  Interval type "range" + Lower limit + High limit - do not give the range its
  own row. Its "Median" row is a separate row (Value + Summary statistics
  "median", nothing else). So one such parameter with N/Range/Mean±SD/Median
  becomes exactly 2 output rows, never 3. When the table also has group-header
  rows splitting this into several such blocks (see the Population rule
  above), restart this column-by-column reading for each block - each block
  gets its own Subject N, but the same Population value throughout.
- Use only the table below. Do not add data from anywhere else.
- If no row in the table qualifies, output the header row and nothing else.
- Leave a cell empty when the table does not report it. Never write N/A, NA,
  or "not reported".
- Output the header row, then the data rows.
- **Every row must have exactly 15 commas.** An empty field is still a field:
  write its comma. A Cmax row for a maternal cohort with no variation and no
  interval is
  `Lorazepam,Lorazepam,blood,maternal,delivery,12,Cmax,12.96,ng/ml,mean,,,,,,`
  - the empty Variation type, Variation value, Interval type, Lower limit,
  High limit and P value each still get their comma. Leave one out and every
  later column shifts by one, which makes the whole row wrong. Count the
  commas before you write the row. This also means never adding a 17th field
  for a comparison-group label (before/after, with/without a condition, an
  age band, ...): that label gets no column at all, per the Population rule
  above - a "before group" cohort's Cavg row is
  `Vancomycin,Vancomycin,blood,neonates,,60,initial concentration,12.9,mg/L,median,,,25th-75th percentiles,11.3,17.0,<0.001`,
  16 fields, 15 commas, with nothing marking it as "before" except its
  Subject N differing from the matching "after" row.
- Output only the CSV. No commentary, no code fences.

TABLE:

{TABLE}

---
