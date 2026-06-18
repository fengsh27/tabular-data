#!/usr/bin/env python3
"""Deterministic map from a pipeline label to the ollama curation skill to trigger.

The identify-pipelines skill's design stage emits a list of pipeline labels (the
values of `extractor.constants.PipelineTypeEnum`). Turning those into the skills to
trigger must NOT be left to the model: the naming is irregular (`pk_summary` ->
`pk-summary-curation`, but `pk_drug_summary` -> `pk-drug-summary`), so a string
transform would silently misroute. This table is the single source of truth.

CLI: turn a selection into the `selected_pipelines.json` artifact.

    python pipeline_skill_map.py --pmid 12345678 --paper-type Both \
        pk_summary pe_study_outcome > selected_pipelines.json
"""
import argparse
import json
import sys

# label (PipelineTypeEnum value) -> ollama skill name to trigger next.
PIPELINE_TO_SKILL = {
    "pk_summary": "pk-summary-curation",
    "pk_individual": "pk-individual-curation",
    "pk_specimen_summary": "pk-specimen-summary",
    "pk_specimen_individual": "pk-specimen-individual",
    "pk_drug_summary": "pk-drug-summary",
    "pk_drug_individual": "pk-drug-individual",
    "pk_population_summary": "pk-population-summary",
    "pk_population_individual": "pk-population-individual",
    "pe_study_info": "pe-study-info",
    "pe_study_outcome": "pe-study-outcome",
}

# domain of each label, for the paper-type gate.
PK_LABELS = {k for k in PIPELINE_TO_SKILL if k.startswith("pk_")}
PE_LABELS = {k for k in PIPELINE_TO_SKILL if k.startswith("pe_")}


def resolve(labels):
    """Map pipeline labels -> [{pipeline, skill}], preserving order, de-duped.

    Raises KeyError on an unknown label (a misspelled / hallucinated pipeline).
    """
    seen = set()
    out = []
    for label in labels:
        label = label.strip()
        if not label or label in seen:
            continue
        if label not in PIPELINE_TO_SKILL:
            raise KeyError(
                f"unknown pipeline label {label!r}; valid: {sorted(PIPELINE_TO_SKILL)}"
            )
        seen.add(label)
        out.append({"pipeline": label, "skill": PIPELINE_TO_SKILL[label]})
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
