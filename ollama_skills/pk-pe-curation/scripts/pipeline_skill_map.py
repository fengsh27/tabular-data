#!/usr/bin/env python3
"""Deterministic map from a legacy pipeline label to its curation skill folder.

The routing skill's design stage emits a list of pipeline labels (the values of
`extractor.constants.PipelineTypeEnum`). Turning those into the skills to invoke
must NOT be left to the model: the naming is irregular (`pk_summary` ->
`pk-summary-curation`, but `pk_drug_summary` -> `pk-drug-summary`), so a string
transform would silently misroute. This table is the single source of truth, and
`test_pipeline_skill_map.py` asserts it covers every pipeline and every target
folder exists.

CLI: turn a selection into the `selected_pipelines.json` dispatch artifact.

    python pipeline_skill_map.py --pmid 12345678 --paper-type Both \
        pk_summary pe_study_outcome > selected_pipelines.json
"""
import argparse
import json
import sys

# label (PipelineTypeEnum value) -> sub-procedure dir under the bundle (relative to
# the skill root). The orchestrator follows <dir>/procedure.md for each selection.
PIPELINE_TO_PROCEDURE = {
    "pk_summary": "pipelines/pk-summary-curation",
    "pk_individual": "pipelines/pk-individual-curation",
    "pk_specimen_summary": "pipelines/pk-specimen-summary",
    "pk_specimen_individual": "pipelines/pk-specimen-individual",
    "pk_drug_summary": "pipelines/pk-drug-summary",
    "pk_drug_individual": "pipelines/pk-drug-individual",
    "pk_population_summary": "pipelines/pk-population-summary",
    "pk_population_individual": "pipelines/pk-population-individual",
    "pe_study_info": "pipelines/pe-study-info",
    "pe_study_outcome": "pipelines/pe-study-outcome",
}


def resolve(labels):
    """Map pipeline labels -> [{pipeline, procedure}], preserving order, de-duped.

    Raises KeyError on an unknown label (a misspelled / hallucinated pipeline).
    """
    seen = set()
    out = []
    for label in labels:
        label = label.strip()
        if not label or label in seen:
            continue
        if label not in PIPELINE_TO_PROCEDURE:
            raise KeyError(
                f"unknown pipeline label {label!r}; valid: {sorted(PIPELINE_TO_PROCEDURE)}"
            )
        seen.add(label)
        out.append({"pipeline": label, "procedure": PIPELINE_TO_PROCEDURE[label]})
    return out


def build_selection(pmid, paper_type, labels):
    return {
        "pmid": pmid,
        "paper_type": paper_type,
        "selected": resolve(labels),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("labels", nargs="*", help="selected pipeline labels (PipelineTypeEnum values)")
    ap.add_argument("--pmid", default=None)
    ap.add_argument("--paper-type", default=None, help="PK / PE / Both / Neither")
    ap.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    args = ap.parse_args(argv)

    try:
        selection = build_selection(args.pmid, args.paper_type, args.labels)
    except KeyError as e:
        sys.exit(str(e))

    text = json.dumps(selection, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
