# Stage 01 — Drug / Analyte / Specimen extraction

This file is loaded by `SKILL.md` during the PK individual curation procedure.

## What you are extracting
The set of unique **[Drug name, Analyte, Specimen]** combinations described by
the PK table. This set defines how many distinct drug/measurement contexts the
rest of the curation must account for.

## Inputs (read from the scratch directory)
- `00_markdown_table.md` — the PK table in markdown (Stage 0 output).
- `inputs.md` — the caption, footnotes, and paper title, verbatim. If the
  paper title is absent, proceed without it and rely on the table and caption.

Read these files now; do not rely on the table text remaining in the
conversation, which may have been summarized.

## Definitions
- **Drug name** — the drug administered in the study.
- **Analyte** — the substance actually measured. It may be the parent drug, a
  metabolite, or a co-administered drug whose levels were tracked. Enter only
  the substance name; qualifiers like "free" or "total" do not belong here.
- **Specimen** — the biological sample type (plasma, serum, whole blood, urine,
  CSF, …).

## Procedure
1. Read the table, caption, and title together.
2. Determine every unique [Drug name, Analyte, Specimen] combination present.
   A single table often contains several (e.g. parent drug + metabolite, or
   plasma + urine).
3. Verify each element actually appears in the table, caption, or title before
   including it. Do not invent specimens or analytes.
4. If an element is missing, first try to infer it from context — related
   rows, the caption, or well-established PK knowledge for the named drug. Use
   `"N/A"` only as a last resort.
5. If none of the three elements appear in the table or caption, infer them
   from the paper title.
6. If the table genuinely contains no drug information at all (for example a
   demographics-only table routed here by mistake), record the single
   combination `["N/A", "N/A", "N/A"]` and note the anomaly to the user.

## Reasoning then answer
Work through your reasoning explicitly first — state which rows, footnotes, or
title fragments justify each combination — and only then produce the result.

## Output of this stage
Produce a markdown table with exactly these columns, one row per unique
combination:

```
| Drug name | Analyte | Specimen |
|-----------|---------|----------|
| Lorazepam | Lorazepam | Plasma |
| Lorazepam | Lorazepam | Urine  |
```

Before continuing, sanity-check the table:
- at least one row,
- every row has all three cells filled (a literal `N/A` counts as filled).

If that check fails, redo this stage once, paying attention to what was wrong
the first time.

Then **write the table to `01_drug_table.md`** in the scratch directory. Later
stages read that file, not this message — so write the complete, exact table.

## Worked example

**Table** (truncated):

```
| Parameter | Healthy (n=12) | Cirrhotic (n=8) |
|-----------|----------------|-----------------|
| AUC0-∞ (ng·h/mL) | 1240 ± 180 | 2890 ± 410 |
| Cmax (ng/mL)     | 95 ± 14    | 142 ± 22    |
```

**Caption**: "Plasma pharmacokinetics of lorazepam after a single 2 mg oral
dose in healthy volunteers and patients with hepatic cirrhosis."

**Paper title**: "Pharmacokinetics of lorazepam in cirrhosis"

**Reasoning**: the caption names lorazepam as the administered drug, measured
in plasma; there is no metabolite column, so the analyte equals the drug; only
one specimen (plasma) is mentioned.

**Result**:

```
| Drug name | Analyte   | Specimen |
|-----------|-----------|----------|
| Lorazepam | Lorazepam | Plasma   |
```
