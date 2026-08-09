#!/usr/bin/env python3
"""Compact per-table digest of a prepared paper, for the identification gate.

Stage 1 of pk-pe-route needs to know whether a paper's tables report an analyte
concentration or a PK parameter. The prepared assets do not offer that cheaply:

  * `table_<n>.md`  holds ONLY the caption + footnotes -- no table rows at all.
  * `table_<n>.html` holds the whole table, pretty-printed with full attributes,
    and runs 4-40x larger than the text it contains.

So "read the table headers" cannot be satisfied by reading the small file, and
reading the big one pulls the entire table body into a gate meant to be cheap.
This script closes that gap: it emits, per table, the caption + footnotes (from
the .md) and the first few rows (parsed out of the .html), which is where the PK
signal actually lives.

Both header rows AND leading data rows are emitted on purpose. In this corpus the
parameter names are frequently row labels rather than column headers -- a table
whose header is only "Variable | Group A | Group B" can have "Mean serum fentanyl
concentration (nmol/L)" as its first data row. Header-only extraction misses those.

Cells are emitted as flat text; colspan/rowspan are NOT expanded, so a spanned
header may look narrower than it renders. That is fine for a routing gate -- use
the full `table_<n>.html` when the table's exact grid matters.

Standard library only (html.parser); no BeautifulSoup, so pk-pe-route stays
dependency-free.

Usage:
    python table_digest.py <prepared-paper-dir>
    python table_digest.py <prepared-paper-dir> --max-rows 5
    python table_digest.py <prepared-paper-dir> --max-rows 3 --max-chars 300

Exit code: 0 on success (including a paper with no tables), 1 on a bad path.
"""

import argparse
import glob
import json
import os
import re
import sys
from html.parser import HTMLParser


class _TableRows(HTMLParser):
    """Collect the leading rows of the first <table> in a prepared table_<n>.html.

    Tolerant of the malformed markup publisher HTML routinely carries: an
    unclosed <td> is closed by the next cell or row, an unclosed <tr> by the next
    row. Nested tables are tracked by depth so an inner table cannot end the
    outer one early.
    """

    CELL_TAGS = ("th", "td")

    def __init__(self, max_rows):
        super().__init__(convert_charrefs=True)
        self.max_rows = max_rows
        self.rows = []
        self._depth = 0
        self._row = None
        self._cell = None

    # -- helpers ---------------------------------------------------------
    def _close_cell(self):
        if self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
        self._cell = None

    def _close_row(self):
        self._close_cell()
        if self._row is not None and any(c for c in self._row):
            self.rows.append(self._row)
        self._row = None

    @property
    def done(self):
        return len(self.rows) >= self.max_rows

    # -- parser hooks ----------------------------------------------------
    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        if tag == "table":
            self._depth += 1
        elif tag == "tr" and self._depth:
            self._close_row()          # previous <tr> never closed
            self._row = []
        elif tag in self.CELL_TAGS and self._depth:
            self._close_cell()         # previous cell never closed
            if self._row is None:      # cell outside any <tr>
                self._row = []
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            if self._depth:
                self._depth -= 1
            if not self._depth:
                self._close_row()
        elif tag == "tr":
            self._close_row()
        elif tag in self.CELL_TAGS:
            self._close_cell()

    def handle_data(self, data):
        if self._cell is not None and not self.done:
            self._cell.append(data)


def _clip(text, limit):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def parse_md(path, max_chars):
    """Return (caption, footnotes) from a prepared table_<n>.md.

    Layout written by prepare_paper.py: `# Table n`, blank, caption, blank,
    `**Footnotes:**`, blank, `- <footnote>` lines. Any part may be absent.
    """
    if not os.path.isfile(path):
        return "", ""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    body = re.sub(r"^#\s*Table\s*\d+\s*$", "", text, count=1, flags=re.M)
    head, sep, foot = body.partition("**Footnotes:**")
    caption = _clip(head, max_chars)
    footnotes = _clip(re.sub(r"^\s*-\s*", "", foot, flags=re.M), max_chars) if sep else ""
    return caption, footnotes


def parse_html_rows(path, max_rows, max_chars):
    """Return up to max_rows leading rows of the table, each as a clipped string."""
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        parser = _TableRows(max_rows)
        try:
            parser.feed(fh.read())
            parser.close()
        except Exception as exc:  # malformed beyond html.parser's tolerance
            print(f"  (row parse failed: {exc})", file=sys.stderr)
    return [_clip(" | ".join(r), max_chars) for r in parser.rows[:max_rows]]


def table_files(paper_dir):
    """Return [(label, md_path, html_path)] in table order, manifest first."""
    manifest = os.path.join(paper_dir, "manifest.json")
    if os.path.isfile(manifest):
        try:
            with open(manifest, encoding="utf-8") as fh:
                data = json.load(fh)
            return [
                (
                    "table_%s" % t.get("n"),
                    os.path.join(paper_dir, t.get("md") or ""),
                    os.path.join(paper_dir, t.get("html") or ""),
                )
                for t in data.get("tables", [])
            ]
        except (ValueError, OSError):
            pass  # fall through to globbing
    out = []
    for html in sorted(
        glob.glob(os.path.join(paper_dir, "table_*.html")),
        key=lambda p: int(re.search(r"table_(\d+)\.html$", p).group(1))
        if re.search(r"table_(\d+)\.html$", p)
        else 0,
    ):
        label = os.path.basename(html)[: -len(".html")]
        out.append((label, os.path.join(paper_dir, label + ".md"), html))
    return out


def digest(paper_dir, max_rows, max_chars):
    pmid = os.path.basename(os.path.normpath(paper_dir))
    tables = table_files(paper_dir)
    lines = []
    if not tables:
        lines.append("# Table digest for %s: no tables." % pmid)
        return "\n".join(lines) + "\n"
    lines.append("# Table digest for %s (%d table(s))" % (pmid, len(tables)))
    lines.append(
        "# caption + footnotes from table_<n>.md; leading rows from table_<n>.html"
    )
    for label, md, html in tables:
        caption, footnotes = parse_md(md, max_chars)
        rows = parse_html_rows(html, max_rows, max_chars)
        lines.append("")
        lines.append("## %s" % label)
        lines.append("Caption: %s" % (caption or "(none)"))
        if footnotes:
            lines.append("Footnotes: %s" % footnotes)
        if rows:
            for i, row in enumerate(rows, 1):
                lines.append("Row %d: %s" % (i, row))
        else:
            lines.append("Rows: (none parsed — open %s.html directly)" % label)
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paper_dir", help="prepared paper dir, e.g. .paper_assets/<pmid>/")
    ap.add_argument("--max-rows", type=int, default=3,
                    help="leading rows to show per table (default: 3)")
    ap.add_argument("--max-chars", type=int, default=300,
                    help="max characters per caption / footnote / row (default: 300)")
    ap.add_argument("--out", default=None, help="write here (default: stdout)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.paper_dir):
        sys.exit("not a directory: %s" % args.paper_dir)

    text = digest(args.paper_dir, max(1, args.max_rows), max(20, args.max_chars))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
