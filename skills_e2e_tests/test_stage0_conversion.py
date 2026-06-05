"""Deterministic regression test for Stage 0 of the pk-summary-curation skill.

Stage 0 (HTML -> Markdown) is the only fully deterministic stage, so it is the
only one we can assert byte-exactly in CI. It guards against regressions in the
shared bundled converter (skills/curation-common/scripts/html_to_markdown_table.py).

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
    "curation-common",
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
