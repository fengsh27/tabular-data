#!/usr/bin/env python3
"""Run the pk-individual simple prompt over gpt-4o for the tables the meta
files (benchmark/data/source/pk-individual/{pmid}.json, see the pk-individual
baseline-vs-source comparison) marked as included in the baseline.

For each pmid:
  1. Read benchmark/data/source/pk-individual/{pmid}.json for the list of
     table indices with included_in_baseline == true.
  2. Pull each such <table> out of benchmark/data/source/pk-individual/{pmid}.html
     (by document order, matching the index the meta file was built with) and
     convert it to Markdown.
  3. Prompt gpt-4o (Azure, via extractor.request_openai.get_openai(), reading
     model/deployment from .env) with prompts/simple_prompts/pk_individual.md.
  4. Parse the CSV response with scripts/simple_prompt/_common.py and
     concatenate all tables' rows for that pmid.

Intermediate artifacts (per-table markdown + raw model output) are written
under benchmark/data/source/pk-individual/scratch/{pmid}/table{index}/.
The final combined CSV per paper goes to
benchmark/data/pk-individual/{version}/{pmid}.csv.

Usage:
    python scripts/simple_prompt/run_gpt4o_pk_individual.py \
        --meta-dir benchmark/data/source/pk-individual \
        --out benchmark/data/pk-individual/2026-9-8

    Add --dry-run to render prompts/markdown without calling the model.
    Add --resume to skip tables whose raw_response.csv already exists.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(
    REPO_ROOT, "skills", "pk-pe-curation", "curation-common", "scripts"
))

import _common as common  # pk-individual COLS / parse_csv / write_csv
from html_to_markdown_table import single_html_table_to_markdown

PROMPT_PATH = os.path.join(REPO_ROOT, "prompts", "simple_prompts", "pk_individual.md")


def load_prompt(path: str) -> str:
    """Take the block between the first two standalone '---' rules."""
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"(?m)^---\s*$", text)
    body = parts[1] if len(parts) >= 3 else text
    return body.strip("\n")


def extract_table_html(html_path: str, index: int) -> str:
    """Pull the outer HTML of the <index>-th <table> tag, document order.

    Must use the same parser (html5lib) as the meta-generation step so the
    index lines up with what benchmark/data/source/pk-individual/{pmid}.json
    recorded.
    """
    from bs4 import BeautifulSoup

    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html5lib")
    tables = soup.find_all("table")
    return str(tables[index])


def call_gpt4o(client, prompt: str) -> str:
    from langchain_core.messages import HumanMessage

    resp = client.invoke([HumanMessage(content=prompt)])
    return resp.content


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", default="benchmark/data/source/pk-individual",
                     help="dir with {pmid}.html + {pmid}.json meta files")
    ap.add_argument("--out", required=True,
                     help="e.g. benchmark/data/pk-individual/2026-9-8")
    ap.add_argument("--pmids", default=None, help="comma list, default: all in --meta-dir")
    ap.add_argument("--resume", action="store_true", help="skip tables already called")
    ap.add_argument("--dry-run", action="store_true",
                     help="write prompts/markdown only, do not call the model")
    args = ap.parse_args()

    if not args.dry_run:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
        from extractor.request_openai import get_openai
        client = get_openai()
    else:
        client = None

    template = load_prompt(PROMPT_PATH)
    if "{TABLE}" not in template:
        print(f"[error] {PROMPT_PATH} has no {{TABLE}} placeholder", file=sys.stderr)
        return 2

    if args.pmids:
        pmids = [p.strip() for p in args.pmids.split(",") if p.strip()]
    else:
        pmids = sorted(
            os.path.basename(p)[:-5]
            for p in glob.glob(os.path.join(args.meta_dir, "*.json"))
        )

    scratch_root = os.path.join(args.meta_dir, "scratch")
    os.makedirs(args.out, exist_ok=True)
    print(f"papers={len(pmids)} out={args.out}{' [dry-run]' if args.dry_run else ''}")

    manifest = []
    for pmid in pmids:
        meta_path = os.path.join(args.meta_dir, f"{pmid}.json")
        html_path = os.path.join(args.meta_dir, f"{pmid}.html")
        meta = json.load(open(meta_path, encoding="utf-8"))
        included = [t["index"] for t in meta["tables"] if t["included_in_baseline"]]
        print(f"[{pmid}] {len(included)} included table(s): {included}")

        rows = []
        for idx in included:
            tdir = os.path.join(scratch_root, pmid, f"table{idx}")
            os.makedirs(tdir, exist_ok=True)

            md_path = os.path.join(tdir, "00_markdown_table.md")
            table_html = extract_table_html(html_path, idx)
            markdown = single_html_table_to_markdown(table_html)
            open(md_path, "w", encoding="utf-8").write(markdown)

            prompt = template.replace("{PMID}", pmid).replace("{TABLE}", markdown)
            prompt_path = os.path.join(tdir, "prompt.txt")
            open(prompt_path, "w", encoding="utf-8").write(prompt)

            raw_path = os.path.join(tdir, "raw_response.csv")
            if args.dry_run:
                print(f"  table{idx}: wrote markdown + prompt (dry-run)")
                continue

            if args.resume and os.path.exists(raw_path):
                raw = open(raw_path, encoding="utf-8").read()
                note = "cached"
            else:
                try:
                    raw = call_gpt4o(client, prompt)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [error] table{idx}: {exc}")
                    manifest.append({"pmid": pmid, "table": idx, "error": str(exc)})
                    continue
                open(raw_path, "w", encoding="utf-8").write(raw)
                note = "called"

            dropped = []
            parsed = common.parse_csv(raw, pmid=pmid, dropped=dropped)
            rows.extend(parsed)
            lost = f", {len(dropped)} unparsable" if dropped else ""
            print(f"  table{idx}: {len(parsed)} rows ({note}){lost}")
            manifest.append({"pmid": pmid, "table": idx, "rows": len(parsed),
                             "dropped": len(dropped)})

        if not args.dry_run:
            out_path = os.path.join(args.out, f"{pmid}.csv")
            common.write_csv(out_path, rows)
            print(f"[{pmid}] total {len(rows)} rows -> {out_path}")

    if not args.dry_run:
        json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
        print(f"\ndone -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
