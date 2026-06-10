"""Deterministic regression test for the prepare-paper skill.

`prepare_paper.py` is the front door of the curation suite: HTML -> the canonical
input layout (paper_text.md, abstract.md, table_<n>.md/.html, manifest.json). It
is fully deterministic (bs4 only, no model), so its output is asserted byte-exactly
against goldens bundled in prepare_cases/.

Guards: title-as-H1 + reference stripping + [Table N] marker substitution in
paper_text.md; abstract extraction; per-table caption+footnote markdown; and the
manifest's marker<->file index that the routing skill depends on.

Run:  poetry run pytest skills_e2e_tests/test_prepare_paper.py
"""
import importlib.util
import json
import os

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "skills", "pk-pe-curation", "curation-common", "scripts", "prepare_paper.py")
CASE = os.path.join(HERE, "prepare_cases")


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_paper", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = load_script()


def _golden(name):
    with open(os.path.join(CASE, name), encoding="utf-8") as fh:
        return fh.read()


def _run(tmp_path):
    """Run the converter on the bundled fixture; return the output dir."""
    src = os.path.join(CASE, "sample_paper.html")
    report = pp.process_paper(src, str(tmp_path), dry_run=False)
    assert report["n_tables"] == 2
    return os.path.join(str(tmp_path), "sample_paper")


def _read(out_dir, name):
    with open(os.path.join(out_dir, name), encoding="utf-8") as fh:
        return fh.read()


def test_all_expected_files_written(tmp_path):
    out = _run(tmp_path)
    for name in ("paper_text.md", "abstract.md", "table_1.md", "table_1.html",
                 "table_2.md", "table_2.html", "manifest.json"):
        assert os.path.isfile(os.path.join(out, name)), f"missing {name}"


def test_paper_text_matches_golden(tmp_path):
    out = _run(tmp_path)
    assert _read(out, "paper_text.md") == _golden("expected_paper_text.md")


def test_abstract_matches_golden(tmp_path):
    out = _run(tmp_path)
    assert _read(out, "abstract.md") == _golden("expected_abstract.md")


def test_table_md_matches_golden(tmp_path):
    out = _run(tmp_path)
    assert _read(out, "table_1.md") == _golden("expected_table_1.md")
    assert _read(out, "table_2.md") == _golden("expected_table_2.md")


def test_paper_text_has_title_and_markers_no_references(tmp_path):
    out = _run(tmp_path)
    text = _read(out, "paper_text.md")
    assert text.startswith("# Pharmacokinetics of DrugX in Pregnant Patients")
    assert "[Table 1]" in text and "[Table 2]" in text
    # references section is stripped
    assert "References" not in text
    assert "Smith J" not in text
    # table grids are not inlined into the prose
    assert "AUC" not in text


def test_table_html_is_section_with_grid(tmp_path):
    out = _run(tmp_path)
    html = _read(out, "table_1.html")
    assert "<section" in html
    assert "<table" in html
    assert "AUC" in html  # the grid lives in the .html, not paper_text.md


def test_manifest_index(tmp_path):
    out = _run(tmp_path)
    manifest = json.loads(_read(out, "manifest.json"))
    assert manifest["pmid"] == "sample_paper"
    assert manifest["title"] == "Pharmacokinetics of DrugX in Pregnant Patients"
    assert manifest["n_tables"] == 2
    assert manifest["has_abstract"] and manifest["has_body"]
    assert manifest["tables"] == [
        {"n": 1, "marker": "[Table 1]", "md": "table_1.md", "html": "table_1.html"},
        {"n": 2, "marker": "[Table 2]", "md": "table_2.md", "html": "table_2.html"},
    ]
    # every marker in the manifest is present in paper_text.md (splice contract)
    text = _read(out, "paper_text.md")
    for t in manifest["tables"]:
        assert t["marker"] in text


def test_dry_run_writes_nothing(tmp_path):
    src = os.path.join(CASE, "sample_paper.html")
    report = pp.process_paper(src, str(tmp_path), dry_run=True)
    assert report["n_tables"] == 2
    assert not os.path.exists(os.path.join(str(tmp_path), "sample_paper"))
