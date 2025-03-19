import re

def display_md_table(md_table):
    """
    Adds labels to each row in a Markdown table.

    The first row is prefixed with "col:", and subsequent rows are prefixed with "row 1:", "row 2:", etc.

    :param md_table: Markdown table as a string
    :return: Modified Markdown table with labeled rows
    """
    lines = md_table.strip().split('\n')
    if len(lines) < 2:
        return md_table  # Not enough lines to form a valid table

    labeled_lines = []
    for i, line in enumerate(lines):
        if i == 0:
            headers = line.split('|')
            headers = [f'"{header.strip()}"' for header in headers if header.strip()]
            labeled_lines.append(f"col: | {' | '.join(headers)} |")
        elif i == 1 and re.match(r'\|\s*-+\s*\|', line):
            labeled_lines.append(line)  # Keep separator unchanged
        else:
            labeled_lines.append(f"row {i - 2}: {line}")

    return '\n'.join(labeled_lines)