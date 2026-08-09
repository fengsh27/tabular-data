#!/usr/bin/env python3
"""Compact per-table digest of a prepared paper, for the identification gate.

Stage 1 of pk-pe-route needs to know whether a paper's tables report an analyte
concentration or a PK parameter, cheaply, on every paper. `pk-pe-prepare` writes
each table to `table_<n>.md` as caption + footnotes + the full Markdown table;
this script trims that to a fixed number of leading rows so a paper with several
large tables still costs little to screen.

Both the header row AND leading data rows are kept on purpose. In this corpus
parameter names are frequently row labels rather than column headers -- a table
whose header is only "Variable | Group A | Group B" can have "Mean serum fentanyl
concentration (nmol/L)" as its first data row, and a subject-labelled first column
("Volunteer", "Patient", or bare 1, 2, 3) is what distinguishes individual-level
data from summary data. Header-only extraction misses both.

Reads the prepared Markdown only -- no HTML parsing, no dependencies. When a
table is truncated the digest says so, with the full row count from manifest.json,
so the reader knows to open `table_<n>.md` in full if it needs the rest.

Usage:
    python table_digest.py <prepared-paper-dir>
    python table_digest.py <prepared-paper-dir> --max-rows 6
    python table_digest.py <prepared-paper-dir> --max-rows 0     # no truncation

Exit code: 0 on success (including a paper with no tables), 1 on a bad path.
"""

import argparse
import glob
import json
import os
import re
import sys

TABLE_MARKER = "**Table:**"
FOOTNOTE_MARKER = "**Footnotes:**"
SEPARATOR_RE = re.compile(r"^\|\s*:?-{2,}")


def split_table_md(text):
    """Split a prepared table_<n>.md into (caption, footnotes, table_lines).

    Layout written by prepare_paper.py: `# Table n`, caption, `**Footnotes:**`
    with `- ` bullets, `**Table:**` with a Markdown table. Any part may be absent.
    """
    body = re.sub(r"^#\s*Table\s*\d+\s*$", "", text, count=1, flags=re.M)
    head, _, table = body.partition(TABLE_MARKER)
    caption, sep, foot = head.partition(FOOTNOTE_MARKER)
    footnotes = re.sub(r"^\s*-\s*", "", foot, flags=re.M) if sep else ""
    table_lines = [ln for ln in table.strip().splitlines() if ln.strip()]
    return " ".join(caption.split()), " ".join(footnotes.split()), table_lines


def clip(text, limit):
    text = " ".join(text.split())
    return text if limit <= 0 or len(text) <= limit else text[: limit - 1].rstrip() + "…"


def table_files(paper_dir):
    """Return [(label, md_path, n_rows)] in table order, manifest first."""
    manifest = os.path.join(paper_dir, "manifest.json")
    if os.path.isfile(manifest):
        try:
            with open(manifest, encoding="utf-8") as fh:
                data = json.load(fh)
            return [
                ("table_%s" % t.get("n"),
                 os.path.join(paper_dir, t.get("md") or ""),
                 t.get("n_rows"))
                for t in data.get("tables", [])
            ]
        except (ValueError, OSError):
            pass  # fall through to globbing

    def _n(path):
        m = re.search(r"table_(\d+)\.md$", path)
        return int(m.group(1)) if m else 0

    return [
        (os.path.basename(p)[: -len(".md")], p, None)
        for p in sorted(glob.glob(os.path.join(paper_dir, "table_*.md")), key=_n)
    ]


def digest(paper_dir, max_rows, max_chars):
    pmid = os.path.basename(os.path.normpath(paper_dir))
    tables = table_files(paper_dir)
    if not tables:
        return "# Table digest for %s: no tables.\n" % pmid

    out = ["# Table digest for %s (%d table(s))" % (pmid, len(tables))]
    if max_rows > 0:
        out.append("# showing up to %d row(s) per table; "
                   "read table_<n>.md for the rest" % max_rows)
    for label, md_path, n_rows in tables:
        out.append("")
        out.append("## %s" % label)
        if not os.path.isfile(md_path):
            out.append("(missing %s)" % os.path.basename(md_path))
            continue
        with open(md_path, encoding="utf-8") as fh:
            caption, footnotes, lines = split_table_md(fh.read())
        out.append("Caption: %s" % (clip(caption, max_chars) or "(none)"))
        if footnotes:
            out.append("Footnotes: %s" % clip(footnotes, max_chars))
        if not lines:
            out.append("Rows: (no table found in %s)" % os.path.basename(md_path))
            continue
        shown = lines if max_rows <= 0 else lines[:max_rows]
        out.extend(clip(ln, max_chars) for ln in shown)
        # the separator line is structure, not data
        data_shown = sum(1 for ln in shown if not SEPARATOR_RE.match(ln.strip())) - 1
        total = n_rows if isinstance(n_rows, int) else len(lines) - 2
        if total > max(0, data_shown):
            out.append("… %d more data row(s); full table in %s"
                       % (total - max(0, data_shown), os.path.basename(md_path)))
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paper_dir", help="prepared paper dir, e.g. .paper_assets/<pmid>/")
    ap.add_argument("--max-rows", type=int, default=4,
                    help="leading table lines to show per table, 0 for all "
                         "(default: 4 — header, separator, 2 data rows)")
    ap.add_argument("--max-chars", type=int, default=300,
                    help="max characters per caption / footnote / row (default: 300)")
    ap.add_argument("--out", default=None, help="write here (default: stdout)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.paper_dir):
        sys.exit("not a directory: %s" % args.paper_dir)

    text = digest(args.paper_dir, args.max_rows, max(20, args.max_chars))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
