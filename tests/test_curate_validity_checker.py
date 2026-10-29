"""Regression tests for extractor.curate_agent.validity_checker (Phase 1).

Exercises the two already-diagnosed `simple_qwen38` failures from
docs/SIMPLE_MODE_QWEN38_ISSUES.md - PMID 30825333's column-shift row and PMID
34183327's zero-row table - as the acceptance test for "does the checker
actually catch what it's meant to." Fixtures are reconstructed from the
issue doc's own quoted data, not copies of the original run (which lived on
scratch and is gone), but they reproduce the exact bug shape.
"""

from extractor.curate_agent import validity_checker as vc

PK_SUMMARY_HEADER = (
    "Drug name,Analyte,Specimen,Population,Pregnancy stage,Subject N,"
    "Parameter type,Value,Unit,Summary statistics,Variation type,"
    "Variation value,Interval type,Lower limit,High limit,P value"
)

SOURCE_MD = """\
| Drug | Parameter | N | Value | Unit |
| --- | --- | --- | --- | --- |
| Ketamine | Total area under curve | 19 | 91 | Mg/min/L |
"""


def test_column_shift_row_is_flagged_and_escalates(tmp_path):
    # The exact shifted row quoted in docs/SIMPLE_MODE_QWEN38_ISSUES.md Issue 1:
    # the model never wrote "Subject N", so every later field slid one column
    # over. Field count still matches the 16-column schema.
    csv_path = tmp_path / "30825333_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,,children,,Total area under curve,91,"
        "Mg/min/L,mean,SD,30,,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        label_columns=["Analyte", "Specimen", "Population", "Parameter type"],
    )

    assert report["tier1"]["total_rows"] == 1
    assert report["tier1"]["dropped_rows"] == 1  # row scores <= 0 under _fits()
    assert report["error_rate"] == 1.0
    assert report["escalate"] is True


def test_zero_row_table_is_flagged_and_escalates(tmp_path):
    # docs/SIMPLE_MODE_QWEN38_ISSUES.md Issue 2: table_2, the paper's actual PK
    # data table, produced zero rows (the drug-as-subheader nesting confused
    # the model). Tier 2 must not run on a zero-row table - there is nothing
    # to check provenance on.
    csv_path = tmp_path / "34183327_combined.csv"
    csv_path.write_text(PK_SUMMARY_HEADER + "\n", encoding="utf-8")
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        label_columns=["Analyte", "Specimen", "Population", "Parameter type"],
    )

    assert report["tier1"]["total_rows"] == 0
    assert report["tier2"] is None
    assert report["error_rate"] == 1.0
    assert report["escalate"] is True


def test_clean_table_does_not_escalate(tmp_path):
    # A well-formed table with no structural or attribution problems should
    # not trip the gate - this is the "stay quiet on known-good papers" half
    # of the acceptance bar, not just "flags known-bad ones."
    csv_path = tmp_path / "clean_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,Plasma,children,,19,Total area under curve,91,"
        "Mg/min/L,mean,SD,30,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        label_columns=["Analyte", "Specimen", "Population", "Parameter type"],
        extra_dropped_rows=0,  # simulating Arm A's manifest.json, which already says 0
    )

    assert report["tier1"]["total_rows"] == 1
    assert report["tier1"]["dropped_rows"] == 0
    assert report["tier2"]["unsupported"] == 0
    assert report["error_rate"] == 0.0
    assert report["escalate"] is False
