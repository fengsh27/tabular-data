---
name: pk-pe-prepare
description: Prepare a PK/PE paper for curation: convert a publisher HTML file or a JATS/PMC XML file into the canonical input layout (paper_text.md with references stripped and tables replaced by [Table N] markers, abstract.md, per-table table_<n>.md / table_<n>.html, and manifest.json). Use this first, before the curation skills, when the user has a raw paper file (.html or .xml). Runs standalone or as Stage A of the pk-pe-curation router.
---

# PK/PE Prepare

The deterministic **front door** of the curation suite. It turns one raw paper
file — **HTML** (PMC / Wiley / Elsevier) or **JATS/PMC XML** — into the canonical
input layout every curation skill expects. Format is auto-detected from the file
extension and root element.

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

## Run
```bash
python scripts/prepare_paper.py <paper.html|paper.xml> --out ./.paper_assets
# a directory of .html/.xml files works too:
python scripts/prepare_paper.py <dir> --out ./.paper_assets
python scripts/prepare_paper.py <paper> --dry-run     # report only, write nothing
```
Needs `beautifulsoup4` **only for HTML** input (`pip install -r scripts/requirements.txt`);
the XML path is Python-3 standard library only.

## Output — `./.paper_assets/<pmid>/`
| File | Contents |
|---|---|
| `paper_text.md` | title (H1) + body as Markdown; references stripped; each data table → a `[Table N]` marker |
| `abstract.md` | the abstract as Markdown |
| `table_<n>.md` | table *n*'s caption + footnotes |
| `table_<n>.html` | table *n* as a `<section>` (caption + table + footnotes) |
| `manifest.json` | title, table count, and the `[Table N]` ↔ file mapping |

Tables are numbered by order of appearance. `<pmid>` is the input file's base name.

## Hand off
Point the curation skills (or the `pk-pe-curation` router) at the produced
`<pmid>/` directory:
- **table** skills (`pk-summary-curation`, `pk-individual-curation`,
  `pe-study-outcome`) → `table_<n>.html`;
- **full-text** skills (`pk-drug-*`, `pk-specimen-*`, `pk-population-*`,
  `pe-study-info`) → `paper_text.md` (+ `abstract.md`).

## Scope & notes
- **HTML** uses best-effort PMC / Wiley / Elsevier selectors. **XML** uses the
  NLM/JATS `<article>` schema (`<front>` metadata, `<body>` sections,
  `<table-wrap>` tables, `<ref-list>` references).
- Output is written to a git-ignored scratch dir in the user's project root
  (`./.paper_assets/`), never inside this skill folder.
- If the user pasted raw text instead of a file, place it into the equivalent
  `paper_text.md` / `table_<n>.html` files by hand.
