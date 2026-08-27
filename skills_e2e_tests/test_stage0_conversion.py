"""Deterministic regression test for Stage 0 of the pk-summary-curation skill.

Stage 0 (HTML -> Markdown) is the only fully deterministic stage, so it is the
only one we can assert byte-exactly in CI. It guards against regressions in the
shared bundled converter (skills/pk-pe-curation/curation-common/scripts/html_to_markdown_table.py).

The later, LLM-driven stages (01-03) are evaluated separately against the
semantic oracles in each case directory -- see README.md. They are not asserted
here because their output depends on a model.

Run:  poetry run pytest skills_e2e_tests/test_stage0_conversion.py
"""
import importlib.util
import os

SKILL_CONVERTER = os.path.join(
    os.path.dirname(__file__),
    "..",
    "skills",
    "pk-pe-curation", "curation-common",
    "scripts",
    "html_to_markdown_table.py",
)


def _load_converter():
    spec = importlib.util.spec_from_file_location("skill_converter", SKILL_CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage0_matches_golden(case):
    """The skill's converter must reproduce the case's expected_00 golden exactly."""
    converter = _load_converter()
    produced = converter.single_html_table_to_markdown(case["source_html"]).strip()
    expected = case["oracles"]["stage_00_markdown"].strip()
    assert produced == expected, (
        f"\nStage-0 conversion drifted for case {case['name']}.\n"
        f"--- expected ---\n{expected}\n--- produced ---\n{produced}\n"
    )


# --- Focused rowspan invariants (case 32153014_table_2) ---------------------
# The byte-exact golden above already catches drift, but it reports a whole-table
# diff. These name the two defects that case exists for, so a regression says
# WHAT broke: carried rowspan cells leaking across rows, and data values being
# absorbed into column names.

ROWSPAN_CASE = os.path.join(os.path.dirname(__file__), "cases", "32153014_table_2")


def _convert_rowspan_case():
    converter = _load_converter()
    with open(os.path.join(ROWSPAN_CASE, "source_table.html"), encoding="utf-8") as fh:
        md = converter.single_html_table_to_markdown(fh.read())
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in md.strip().splitlines()
        if line.startswith("|")
    ]
    return rows[0], rows[2:]  # header, data rows (rows[1] is the --- separator)


def test_rowspan_headers_carry_no_data_values():
    """Column names must not absorb cell values from the rows they span over."""
    header, _ = _convert_rowspan_case()
    offenders = [h for h in header if any(ch.isdigit() for ch in h)]
    assert offenders == [], (
        "column names absorbed data values from spanned rows: "
        f"{offenders}\nfull header: {header}"
    )


def test_rowspan_continuation_row_keeps_its_own_patients_values():
    """A row that only supplies one column's second line must carry the rest of
    ITS OWN row's spanned values -- not values left over from another patient."""
    _, data = _convert_rowspan_case()
    by_patient = {}
    for row in data:
        by_patient.setdefault(row[0], []).append(row)

    # Patients 1-3 each occupy two physical rows (the second holds the dosing
    # regimen's second line). Measurement columns must agree within a patient.
    for pid in ("1", "2", "3"):
        rows = by_patient[pid]
        assert len(rows) == 2, f"patient {pid}: expected 2 physical rows, got {len(rows)}"
        first, cont = rows
        assert first[3:] == cont[3:], (
            f"patient {pid}'s continuation row carries different measurements "
            f"than its own data row:\n  data: {first}\n  cont: {cont}"
        )

    # The original bug filled patients 1 and 2 with patient 3's numbers.
    assert by_patient["2"][0][3:] != by_patient["3"][0][3:], (
        "patient 2 and patient 3 have identical measurements -- rowspan "
        "residue from a later row leaked into an earlier one"
    )


# --- Multi-line cell invariants (case 18426260_table_2) ---------------------
# Cells wrapped across source lines used to carry their newline into the row,
# splitting one Markdown row over several physical lines. That produced ragged
# widths and an IndexError deeper in the chain -- which prepare_paper.py
# swallowed, shipping the table to the model as a caption with no grid.

NEWLINE_CASE = os.path.join(os.path.dirname(__file__), "cases", "18426260_table_2")


def test_multiline_cells_do_not_split_rows():
    """Every row must be one physical line, the same width as the header."""
    converter = _load_converter()
    with open(os.path.join(NEWLINE_CASE, "source_table.html"), encoding="utf-8") as fh:
        md = converter.single_html_table_to_markdown(fh.read())

    lines = md.strip().splitlines()
    assert all(line.startswith("|") and line.endswith("|") for line in lines), (
        "a cell's embedded newline split a row across physical lines:\n"
        + "\n".join(repr(l) for l in lines if not (l.startswith("|") and l.endswith("|")))
    )

    widths = {len(line.split("|")) - 2 for line in lines}
    assert len(widths) == 1, f"ragged row widths: {sorted(widths)}"


def test_multiline_cell_reads_as_one_value():
    """A cell wrapped in the source must join into a single readable value."""
    converter = _load_converter()
    with open(os.path.join(NEWLINE_CASE, "source_table.html"), encoding="utf-8") as fh:
        md = converter.single_html_table_to_markdown(fh.read())
    assert "0.7 ± 0.1 (0.6–0.8)" in md, (
        "wrapped cell did not join into one value; got row:\n"
        + next((l for l in md.splitlines() if "S-citalopram" in l), "<not found>")
    )
