# Prepare Paper

The **front door** of the curation-skills suite. It turns one raw paper HTML into
a small, predictable set of files that every downstream skill reads from:

- **`paper_text.md`** — the article title (as an H1) + body, with the reference /
  bibliography section removed and every data table replaced by a `[Table N]`
  marker at its original position. This is the *full text* the full-text curation
  skills and the routing skill consume.
- **`abstract.md`** — the abstract.
- **`table_<n>.md`** — table `<n>`'s caption + footnotes (no grid).
- **`table_<n>.html`** — table `<n>` itself, as a `<section>` (caption + table +
  footnotes). This is the exact input the *table* curation skills want
  (pk-summary-curation, pk-individual-curation, pe-study-outcome).
- **`manifest.json`** — the machine-readable index: paper title, table count, and
  the `[Table N]` ↔ `table_<n>.md` / `table_<n>.html` mapping. The routing skill
  uses this to splice tables back inline when it needs the legacy "full text with
  tables visible" view.

Tables are numbered by order of appearance. This skill **only extracts and
splits** — it does not curate, classify, or interpret the content.

## Inputs you need
1. **A paper HTML file** — a single `<pmid>.html`, or a directory of `<pmid>.html`
   files. The HTML should be the publisher's article page (PMC, Wiley, or
   Elsevier/ScienceDirect markup is handled best-effort).

If the user supplies a PMID, URL, or PDF instead, ask them for the article HTML —
this skill does not fetch or convert PDFs. JATS/PMC **XML** is out of scope for
now (the structure differs); tell the user if they hand you XML.

## Output layout

For each paper, files are written under `<out>/<pmid>/` (default `<out>` is
`./.paper_assets/` in the user's current project / working directory — never
inside the skill folder; it is git-ignored):

```
./.paper_assets/<pmid>/
  paper_text.md
  abstract.md
  table_1.md
  table_1.html
  table_2.md
  table_2.html
  ...
  manifest.json
```

## Procedure

This is a **single deterministic step — never hand-parse the HTML.** Run the
bundled converter:

```bash
python curation-common/scripts/prepare_paper.py <paper.html> --out ./.paper_assets
```

- Pass a single `<pmid>.html` file, or a directory of them.
- Add `--dry-run` to report what would be extracted (title / abstract / body
  presence + table count) without writing files — useful to sanity-check that the
  publisher's markup was recognized before committing the output.
- The script depends only on `beautifulsoup4` (already a project dependency).

After it runs, confirm to the user: the `<pmid>/` directory, the number of tables
extracted, and whether the title / abstract / body were all found (the script
prints a `<-- CHECK` flag when any are missing, which usually means an
unrecognized publisher layout).

## What to do next

Hand the produced files to the appropriate downstream skill:
- **`pk-pe-route`** — to decide which curation pipelines apply (reads
  `paper_text.md` + `manifest.json` + the `table_<n>.md` files).
- **Table skills** (`pk-summary-curation`, `pk-individual-curation`,
  `pe-study-outcome`) — give them the relevant `table_<n>.html`.
- **Full-text skills** (`pk-drug-*`, `pk-specimen-*`, `pk-population-*`,
  `pe-study-info`) — give them `paper_text.md` (and `abstract.md` for context).

## Scope & limitations
- **HTML only.** PMC / Wiley / Elsevier markup is recognized best-effort via
  structural selectors. Unusual layouts may miss the body or a table — the
  `--dry-run` report and the `<-- CHECK` flag surface this.
- **No XML.** JATS/PMC XML has a different structure and is a planned follow-up.
- **Extraction, not curation.** Reference stripping uses id/class heuristics; the
  Markdown body is flattened (headings, paragraphs, lists) for LLM consumption,
  not pixel-perfect fidelity.
