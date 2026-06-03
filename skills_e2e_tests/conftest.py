"""Shared fixtures for skills_e2e_tests.

Discovers every case directory under cases/ so tests can parametrize over them.
A "case" is any directory under cases/ that contains a meta.json.
"""
import json
import os

import pytest

CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")


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
