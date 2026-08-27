#!/usr/bin/env python3
"""Stage 0 of the pk-summary-curation skill: convert a single HTML <table>
into a clean Markdown table that is friendly for an LLM to reason over.

Self-contained: depends only on BeautifulSoup (`pip install beautifulsoup4`)
and the Python standard library. It does NOT import anything from the
surrounding repository, so the skill stays portable.

The conversion logic is vendored from the project's
TabFuncFlow.utils.table_utils so that output matches the legacy pipeline
byte-for-byte. It handles:
  - colspan / rowspan expansion,
  - multi-row header stacking,
  - empty row/column removal,
  - empty-header filling and duplicate-header de-duplication.

Usage:
    python html_to_markdown_table.py path/to/table.html
    cat table.html | python html_to_markdown_table.py
"""

import re
import sys

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "error: BeautifulSoup is required. Install with: pip install beautifulsoup4\n"
    )
    sys.exit(2)


def _place_carried_cells(rowspan_tracker, row_data, col_idx):
    """Emit cells carried down from earlier rows' rowspans, starting at col_idx.

    Returns the next free column index. This must run before *every* real cell,
    not only at the start of the row: a carried cell can sit between two real
    cells -- e.g. a row that supplies only the second line of one non-spanning
    column, while every other column spans down from the row above.
    """
    while col_idx in rowspan_tracker:
        text, remaining = rowspan_tracker[col_idx]
        row_data.append(text)
        if remaining > 1:
            rowspan_tracker[col_idx] = (text, remaining - 1)
        else:
            del rowspan_tracker[col_idx]
        col_idx += 1
    return col_idx


def html_table_to_markdown(html):
    """Convert an HTML table into a Markdown table, handling colspan and rowspan."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return ""
    rows = table.find_all("tr")

    table_matrix = []
    max_cols = 0
    rowspan_tracker = {}  # col_idx -> (text, remaining_rows)

    for row in rows:
        cols = row.find_all(["th", "td"])
        row_data = []
        col_idx = 0

        # A row with no cells of its own is not a header row -- it is a
        # continuation row made up entirely of cells spanning down from above.
        is_header = bool(cols) and all(col.name == "th" for col in cols)

        for col in cols:
            col_idx = _place_carried_cells(rowspan_tracker, row_data, col_idx)

            for sup in col.find_all("sup"):  # Drop superscripts (footnote markers)
                sup.decompose()

            # Collapse internal whitespace: a cell whose text is split over
            # several source lines otherwise carries a newline into the row,
            # which splits one Markdown row across several physical lines.
            text = re.sub(r"\s+", " ", "".join(col.stripped_strings)).strip()
            colspan = int(col.get("colspan", 1) or 1)
            rowspan = int(col.get("rowspan", 1) or 1)

            row_data.extend([text] * colspan)  # Expand colspan cells

            if rowspan > 1:
                for i in range(colspan):
                    rowspan_tracker[col_idx + i] = (text, rowspan - 1)

            col_idx += colspan

        # Carried cells trailing after this row's last real cell.
        col_idx = _place_carried_cells(rowspan_tracker, row_data, col_idx)

        max_cols = max(max_cols, len(row_data))
        table_matrix.append((row_data, is_header))

    # Pad genuinely ragged rows (malformed HTML) with blanks. Padding from
    # rowspan_tracker here read its state *after* the whole table was parsed,
    # so short rows were filled with values left over from unrelated rows.
    for row, _ in table_matrix:
        while len(row) < max_cols:
            row.append("")

    # Identify the contiguous header rows at the top
    header_end_idx = 0
    for i, (_, is_header) in enumerate(table_matrix):
        if is_header:
            header_end_idx = i
        else:
            break

    markdown_rows = ["| " + " | ".join(row) + " |" for row, _ in table_matrix]
    separator = "| " + " | ".join(["---"] * max_cols) + " |"

    return "\n".join(
        markdown_rows[: header_end_idx + 1]
        + [separator]
        + markdown_rows[header_end_idx + 1 :]
    )


def stack_md_table_headers(md_table):
    """Merge multi-line headers by column, joining names with '.'."""
    lines = md_table.strip().split("\n")

    separator_idx = next(
        (i for i, line in enumerate(lines) if re.match(r"\|\s*-+\s*\|", line)), None
    )
    if separator_idx is None or separator_idx == 0:
        return md_table

    header_lines = lines[:separator_idx]
    header_matrix = [re.split(r"\s*\|\s*", line.strip("|")) for line in header_lines]

    stacked_header = [
        col[0] if all(x == col[0] for x in col) else ".".join(filter(None, col))
        for col in zip(*header_matrix)
    ]

    stacked_header_line = "| " + " | ".join(stacked_header) + " |"
    return "\n".join([stacked_header_line] + lines[separator_idx:])


def remove_empty_col_row(md_table):
    """Remove empty rows and columns, retaining columns that have a header."""
    lines = md_table.strip().split("\n")
    if len(lines) < 2:
        return md_table

    headers = lines[0].split("|")[1:-1]
    separator = lines[1]
    data_rows = [line.split("|")[1:-1] for line in lines[2:]]

    headers = [h.strip() for h in headers]
    data_rows = [[cell.strip() for cell in row] for row in data_rows]

    valid_columns = [
        i for i in range(len(headers)) if headers[i] or any(row[i] for row in data_rows)
    ]

    cleaned_headers = "| " + " | ".join(headers[i] for i in valid_columns) + " |"
    cleaned_data_rows = [
        "| " + " | ".join(row[i] for i in valid_columns) + " |"
        for row in data_rows
        if any(row)
    ]

    return "\n".join([cleaned_headers, separator] + cleaned_data_rows)


def fill_empty_headers(md_table):
    """Assign unique 'Unnamed_x' names to empty column headers."""
    lines = md_table.strip().split("\n")
    if len(lines) < 2:
        return md_table

    headers = lines[0].split("|")[1:-1]
    separator = lines[1]

    existing_numbers = set()
    pattern = re.compile(r"Unnamed_(\d+)")
    for header in headers:
        match = pattern.match(header.strip())
        if match:
            existing_numbers.add(int(match.group(1)))

    next_num = 0 if not existing_numbers else max(existing_numbers) + 1

    for i in range(len(headers)):
        headers[i] = headers[i].strip()
        if not headers[i]:
            while next_num in existing_numbers:
                next_num += 1
            headers[i] = f"Unnamed_{next_num}"
            existing_numbers.add(next_num)

    filled_header_line = "| " + " | ".join(headers) + " |"
    return "\n".join([filled_header_line, separator] + lines[2:])


def deduplicate_headers(md_table):
    """Rename duplicate column headers with _0, _1, _2 ... suffixes."""
    lines = md_table.strip().split("\n")
    if len(lines) < 2:
        return md_table

    headers = lines[0].split("|")[1:-1]
    separator = lines[1]

    counts = {}
    for header in headers:
        header = header.strip()
        counts[header] = counts.get(header, 0) + 1

    seen = {}
    new_headers = []
    for header in headers:
        header = header.strip()
        if counts[header] > 1:
            seen[header] = seen.get(header, -1) + 1
            new_headers.append(f"{header}_{seen[header]}")
        else:
            new_headers.append(header)

    deduplicated_header_line = "| " + " | ".join(new_headers) + " |"
    return "\n".join([deduplicated_header_line, separator] + lines[2:])


def single_html_table_to_markdown(html_content):
    """Convert a single HTML <table> to a cleaned Markdown table."""
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    if len(tables) != 1:
        sys.stderr.write(
            f"warning: expected exactly one <table>, found {len(tables)}; "
            "converting the first one.\n"
        )

    html_content = re.sub(r"\xa0", " ", html_content)
    return deduplicate_headers(
        fill_empty_headers(
            remove_empty_col_row(
                stack_md_table_headers(html_table_to_markdown(html_content))
            )
        )
    )


def main(argv):
    if len(argv) > 1:
        with open(argv[1], encoding="utf-8") as fh:
            html = fh.read()
    else:
        html = sys.stdin.read()

    markdown = single_html_table_to_markdown(html)
    if not markdown.strip():
        sys.stderr.write("error: no <table> found in input.\n")
        return 1
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
