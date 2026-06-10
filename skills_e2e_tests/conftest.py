"""Shared fixtures for skills_e2e_tests.

Discovers every case directory under cases/ so tests can parametrize over them.
A "case" is any directory under cases/ that contains a meta.json.

A second family of cases lives under selection_cases/: multi-table inputs that
exercise Stage 0b (select the PK summary tables). Those are loaded separately
because a selection case has many tables and a selection oracle, not a single
source table + per-stage oracles.
"""
import importlib.util
import json
import os

import pytest

CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")
SELECTION_CASES_DIR = os.path.join(os.path.dirname(__file__), "selection_cases")

# The skill's bundled HTML->Markdown converter (Stage 0a), loaded once so tests
# can build the same `00_all_tables.md` input the skill would.
_CONVERTER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "skills",
    "pk-pe-curation", "curation-common",
    "scripts",
    "html_to_markdown_table.py",
)


def _load_converter():
    spec = importlib.util.spec_from_file_location("html_to_markdown_table", _CONVERTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_cases():
    if not os.path.isdir(CASES_DIR):
        return []
    out = []
    for name in sorted(os.listdir(CASES_DIR)):
        case_dir = os.path.join(CASES_DIR, name)
        meta_path = os.path.join(case_dir, "meta.json")
        if os.path.isfile(meta_path):
            out.append((name, case_dir))
    return out


def load_case(case_dir):
    """Return a dict of the case's inputs, oracles, and metadata."""
    with open(os.path.join(case_dir, "meta.json")) as f:
        meta = json.load(f)

    def read(fname):
        path = os.path.join(case_dir, fname)
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    return {
        "name": os.path.basename(case_dir),
        "dir": case_dir,
        "meta": meta,
        "title": read(meta["title_file"]).strip(),
        "caption": read(meta["caption_file"]).strip(),
        "source_html": read(meta["source_table_html"]),
        "oracles": {k: read(v) for k, v in meta["oracles"].items()},
    }


@pytest.fixture(params=discover_cases(), ids=lambda c: c[0])
def case(request):
    _, case_dir = request.param
    return load_case(case_dir)


# --- Stage 0b selection cases (multi-table) ---------------------------------

def discover_selection_cases():
    if not os.path.isdir(SELECTION_CASES_DIR):
        return []
    out = []
    for name in sorted(os.listdir(SELECTION_CASES_DIR)):
        case_dir = os.path.join(SELECTION_CASES_DIR, name)
        meta_path = os.path.join(case_dir, "meta.json")
        if os.path.isfile(meta_path):
            out.append((name, case_dir))
    return out


def load_selection_case(case_dir):
    """Return a selection case's tables, title, and selection oracle."""
    with open(os.path.join(case_dir, "meta.json")) as f:
        meta = json.load(f)

    def read(fname):
        with open(os.path.join(case_dir, fname), encoding="utf-8") as fh:
            return fh.read()

    tables = []
    for t in meta["tables"]:
        tables.append({
            "label": t["label"],
            "caption": t.get("caption", ""),
            "footnote": t.get("footnote", ""),
            "html": read(t["html_file"]),
        })

    return {
        "name": os.path.basename(case_dir),
        "dir": case_dir,
        "meta": meta,
        "title": read(meta["title_file"]).strip(),
        "tables": tables,
        "expected": meta["expected"],
    }


def build_all_tables_md(selection_case):
    """Assemble the Stage-0a `00_all_tables.md` input the skill would build:
    every table converted to markdown, each preceded by its label + caption."""
    converter = _load_converter()
    blocks = []
    for t in selection_case["tables"]:
        md = converter.single_html_table_to_markdown(t["html"])
        cap = t["caption"]
        if t["footnote"]:
            cap = f"{cap}\n{t['footnote']}"
        blocks.append(f"## {t['label']}\nCaption: {cap}\n\n{md}\n")
    return "\n".join(blocks)


@pytest.fixture(params=discover_selection_cases(), ids=lambda c: c[0])
def selection_case(request):
    _, case_dir = request.param
    return load_selection_case(case_dir)
