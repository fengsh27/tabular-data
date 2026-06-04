"""Regression test for Stage 0b — selecting the PK summary tables.

The select-vs-exclude decision is LLM judgment, so this is NOT a byte-exact
assert of the model's choice. What IS deterministic — and worth guarding in
CI — is:

  1. the selection oracle is internally consistent (selected + excluded
     partition every table, are disjoint, and reference only real labels), and
  2. the skill's Stage-0a step can render every input table into the
     `00_all_tables.md` block the selection prompt reads.

Together these catch the structural ways a selection case or the Stage-0a
assembly could rot, without pretending to grade the model offline. The actual
select/exclude judgment is a semantic eval (see README + meta.json rationale).

Run:  poetry run pytest skills_e2e_tests/test_table_selection.py
"""
from conftest import build_all_tables_md


def test_oracle_partitions_all_tables(selection_case):
    labels = {t["label"] for t in selection_case["tables"]}
    selected = set(selection_case["expected"]["selected"])
    excluded = set(selection_case["expected"]["excluded"])

    # every label is accounted for exactly once
    assert selected | excluded == labels
    assert selected & excluded == set()
    # at least one of each so the case actually exercises inclusion AND exclusion
    assert selected, "selection case must select at least one PK table"
    assert excluded, "selection case must exclude at least one non-PK table"


def test_rationale_covers_every_table(selection_case):
    rationale = selection_case["meta"].get("rationale", {})
    labels = {t["label"] for t in selection_case["tables"]}
    assert set(rationale) == labels, "every table needs a documented rationale"


def test_all_tables_md_renders_each_table(selection_case):
    """Stage 0a: each table converts to a non-empty markdown block, labeled."""
    all_md = build_all_tables_md(selection_case)
    for t in selection_case["tables"]:
        assert f"## {t['label']}" in all_md
        # the caption text must accompany the table for the selection prompt
        assert t["caption"].split(";")[0][:30] in all_md
    # the assembled input is non-trivial (has real table rows, not just headers)
    assert all_md.count("| --- |") >= len(selection_case["tables"]) or \
        all_md.count("---") >= len(selection_case["tables"])
