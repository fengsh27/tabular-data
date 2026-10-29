"""Thin, import-based adapter around the existing provenance checker.

`skills/pk-pe-curation/curation-common/scripts/verify_provenance.py` already
implements both checks the design's Tier 2 needs - existence (`check_csv`)
and label-attribution (`check_attribution`, the one that catches column-shift
misattribution, see `docs/VERIFICATION_AGENT_DESIGN.md`) - but only as a CLI
script (`main(argv)`). Its two check functions are already plain, importable
functions; this module just does the file-loading/table-parsing glue
(`load_source_numbers`, `parse_markdown_tables`, `build_groups`) that script's
`main()` does, so a caller can hand it paths and get a findings dict back
without shelling out. `verify_provenance.py` itself is unchanged.
"""

from __future__ import annotations

import importlib
import os
import sys

_VERIFY_PROVENANCE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..",
        "skills", "pk-pe-curation", "curation-common", "scripts",
    )
)


def _verify_provenance():
    if _VERIFY_PROVENANCE_DIR not in sys.path:
        sys.path.insert(0, _VERIFY_PROVENANCE_DIR)
    return importlib.import_module("verify_provenance")


def check_existence(csv_path, source_paths, value_columns=None, skip_columns=None) -> dict:
    """Does every number in `csv_path` appear somewhere in `source_paths`?

    Pure existence check - catches hallucinated values, not column-shift/
    misattribution (a shifted value is still a real number from the source,
    just in the wrong cell). See `check_attribution` for that.
    """
    vp = _verify_provenance()
    source_numbers, _source_raw = vp.load_source_numbers(source_paths)
    findings, checked = vp.check_csv(
        csv_path, source_numbers,
        value_columns=value_columns, skip_columns=skip_columns,
    )
    return {"checked_numbers": checked, "unsupported": len(findings), "findings": findings}


def check_attribution(csv_path, source_paths, label_columns,
                       value_columns=None, skip_columns=None) -> dict:
    """Does every number in `csv_path` sit under its row's best-matching
    source label (Jaccard token overlap against `label_columns`)?

    This is the mechanism that catches misattribution/column-shift: both
    numbers may exist in the source, but each lands under the wrong row.
    `label_columns` must be non-empty (e.g. `["Analyte", "Specimen",
    "Population", "Parameter type"]` for pk_summary).
    """
    if not label_columns:
        raise ValueError("check_attribution requires non-empty label_columns")
    vp = _verify_provenance()
    _source_numbers, source_raw = vp.load_source_numbers(source_paths)
    groups = vp.build_groups(vp.parse_markdown_tables(source_raw))
    findings, checked = vp.check_attribution(
        csv_path, groups, label_columns,
        value_columns=value_columns, skip_columns=skip_columns,
    )
    return {"checked_numbers": checked, "misattributed": len(findings), "findings": findings}
