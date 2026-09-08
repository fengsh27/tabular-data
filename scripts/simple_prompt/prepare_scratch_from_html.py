#!/usr/bin/env python3
"""Build a run_simple_prompt.py scratch tree straight from raw paper HTML,
bypassing the skill's own route/table-selection stages.

pk-individual's simple-prompt benchmark reused a `.pk_individual_scratch` tree
a full skill run had already built (which table-selection stage had already
narrowed to the PK-relevant tables). For a fresh set of papers with no prior
skill run, this script does the deterministic half only: run pk-pe-prepare on
every HTML, then stage EVERY extracted table as its own `table_<n>/` scratch
directory. Nothing here decides which tables are relevant - that's left to the
simple prompt itself (it is expected to return 0 rows for tables that don't
match its schema, e.g. demographics-only or literature-review tables).

    python scripts/simple_prompt/prepare_scratch_from_html.py \
        --html-dir data --assets-out .paper_assets --scratch-out .simple_scratch \
        --prepare-script ollama_skills/pk-pe-prepare/scripts/prepare_paper.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-dir", required=True, help="dir of <pmid>.html files")
    ap.add_argument("--assets-out", required=True, help="pk-pe-prepare output dir")
    ap.add_argument("--scratch-out", required=True,
                    help="run_simple_prompt.py --scratch tree to build")
    ap.add_argument("--prepare-script", required=True,
                    help="path to pk-pe-prepare's prepare_paper.py")
    ap.add_argument("--pmids", default=None, help="comma list; default: every *.html")
    args = ap.parse_args()

    if args.pmids:
        pmids = [p.strip() for p in args.pmids.split(",") if p.strip()]
    else:
        pmids = sorted(
            f[:-5] for f in os.listdir(args.html_dir) if f.endswith(".html")
        )
    if not pmids:
        print(f"[error] no .html files in {args.html_dir}", file=sys.stderr)
        return 2

    os.makedirs(args.assets_out, exist_ok=True)
    os.makedirs(args.scratch_out, exist_ok=True)

    for pmid in pmids:
        html_path = os.path.join(args.html_dir, f"{pmid}.html")
        assets_dir = os.path.join(args.assets_out, pmid)
        print(f"[{pmid}] prepare...")
        subprocess.run(
            [sys.executable, args.prepare_script, html_path, "--out", args.assets_out],
            check=True,
        )
        if not os.path.isdir(assets_dir):
            print(f"  [error] prepare produced no {assets_dir}", file=sys.stderr)
            continue

        manifest_path = os.path.join(assets_dir, "manifest.json")
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        tables = manifest.get("tables", [])
        print(f"  {len(tables)} table(s) in manifest")

        title = (manifest.get("title") or "").strip()
        for t in tables:
            n = t["n"]
            src = os.path.join(assets_dir, t["md"])
            if not os.path.isfile(src):
                print(f"  [warn] table_{n}: no {src}", file=sys.stderr)
                continue
            tdir = os.path.join(args.scratch_out, pmid, f"table_{n}")
            os.makedirs(tdir, exist_ok=True)
            # table_<n>.md already carries caption + footnotes + the markdown
            # table itself, so it doubles as the "00_markdown_table.md" the
            # simple-prompt runner feeds straight into {TABLE} - except it
            # never includes the paper title, which manifest.json holds
            # separately. Many summary tables never repeat the drug name in
            # the table body at all (a single-drug paper states it once, in
            # the title), so prepend it here or the prompt's title-fallback
            # rule has nothing to fall back to.
            body = open(src, encoding="utf-8").read()
            out = f"Paper title: {title}\n\n{body}" if title else body
            open(os.path.join(tdir, "00_markdown_table.md"), "w", encoding="utf-8").write(out)

    print(f"\n[done] scratch tree -> {args.scratch_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
