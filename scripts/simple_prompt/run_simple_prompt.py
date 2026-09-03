#!/usr/bin/env python3
"""Run a single-shot prompt over the tables the pk-individual skill selected.

This is the fast-path arm of the two-tier experiment: one model call per table,
no decomposition, no verification. Point it at the scratch tree left behind by a
skill run so both arms read exactly the same tables.

    python scripts/simple_prompt/run_simple_prompt.py \
        --prompt prompts/simple_prompts/pk_individual.md \
        --scratch .../.pk_individual_scratch \
        --out results/simple_v2 \
        --model qwen3.8:27b

Add --dry-run to render the prompts without contacting the model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

TABLE_FILES = ("00_markdown_table.md", "inputs.md")


def load_prompt(path: str) -> str:
    """Take the block between the first two standalone '---' rules, if present.

    That lets the .md file carry a header and a caveats section around the
    prompt itself without either leaking into the model call.
    """
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"(?m)^---\s*$", text)
    body = parts[1] if len(parts) >= 3 else text
    return body.strip("\n")


def resolve_base_url(explicit: str | None) -> str:
    raw = explicit or os.environ.get("OLLAMA_BASE_URL") or os.environ.get(
        "OLLAMA_HOST"
    ) or "http://localhost:11434"
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    return raw.rstrip("/")


def call_ollama(base_url, model, prompt, temperature, num_ctx, timeout, retries=2):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"{base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode()).get("response", "")
        except Exception as exc:  # noqa: BLE001 - report and retry
            last = exc
            if attempt < retries:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ollama call failed after {retries + 1} tries: {last}")


def find_tables(scratch: str, pmid: str):
    """The table directories the skill selected, with the file to feed the model."""
    root = os.path.join(scratch, pmid)
    found = []
    for name in sorted(os.listdir(root)):
        tdir = os.path.join(root, name)
        if not os.path.isdir(tdir) or not name.startswith("table"):
            continue
        for candidate in TABLE_FILES:
            path = os.path.join(tdir, candidate)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                found.append((name, path, candidate))
                break
        else:
            print(f"  [warn] {pmid}/{name}: no table file, skipped")
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--scratch", required=True, help="a .pk_individual_scratch tree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("SIMPLE_PROMPT_MODEL", "qwen3.8:27b"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--pmids", default=None, help="comma list, or a file of PMIDs")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--resume", action="store_true", help="skip tables already done")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_url = resolve_base_url(args.base_url)
    template = load_prompt(args.prompt)
    if "{TABLE}" not in template:
        print(f"[error] {args.prompt} has no {{TABLE}} placeholder", file=sys.stderr)
        return 2

    if args.pmids and os.path.exists(args.pmids):
        pmids = [l.strip() for l in open(args.pmids) if l.strip()]
    elif args.pmids:
        pmids = [p.strip() for p in args.pmids.split(",") if p.strip()]
    else:
        pmids = sorted(
            d for d in os.listdir(args.scratch)
            if os.path.isdir(os.path.join(args.scratch, d)) and d.isdigit()
        )

    os.makedirs(args.out, exist_ok=True)
    print(f"model={args.model} base_url={base_url} papers={len(pmids)} "
          f"temp={args.temperature}{' [dry-run]' if args.dry_run else ''}")

    all_rows, manifest = [], []
    for pmid in pmids:
        pdir = os.path.join(args.out, pmid)
        os.makedirs(pdir, exist_ok=True)
        tables = find_tables(args.scratch, pmid)
        print(f"[{pmid}] {len(tables)} table(s): {', '.join(t[0] for t in tables)}")

        rows = []
        for tname, tpath, source in tables:
            raw_path = os.path.join(pdir, f"{tname}.raw.txt")
            prompt = template.replace("{PMID}", pmid).replace(
                "{TABLE}", open(tpath, encoding="utf-8").read()
            )

            if args.dry_run:
                open(os.path.join(pdir, f"{tname}.prompt.txt"), "w",
                     encoding="utf-8").write(prompt)
                continue

            if args.resume and os.path.exists(raw_path):
                raw = open(raw_path, encoding="utf-8").read()
                note = "cached"
            else:
                t0 = time.time()
                try:
                    raw = call_ollama(base_url, args.model, prompt,
                                      args.temperature, args.num_ctx, args.timeout)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [error] {tname}: {exc}")
                    manifest.append({"pmid": pmid, "table": tname, "error": str(exc)})
                    continue
                open(raw_path, "w", encoding="utf-8").write(raw)
                note = f"{time.time() - t0:.0f}s"

            dropped = []
            parsed = common.parse_csv(raw, pmid=pmid, dropped=dropped)
            rows.extend(parsed)
            lost = f", {len(dropped)} unparsable" if dropped else ""
            print(f"  {tname}: {len(parsed)} rows ({note}, from {source}){lost}")
            for bad in dropped:
                print(f"    [dropped] {len(bad)} fields: {','.join(bad)[:110]}")
            manifest.append({"pmid": pmid, "table": tname, "source": source,
                             "rows": len(parsed), "dropped": len(dropped)})

        if not args.dry_run:
            common.write_csv(os.path.join(pdir, "combined.csv"), rows)
            # The per-paper file matches the gold's columns exactly; the pooled
            # one needs a PMID or nothing says which paper a row came from.
            all_rows.extend(dict(r, PMID=pmid) for r in rows)
            print(f"[{pmid}] total {len(rows)} rows")

    if not args.dry_run:
        common.write_csv(os.path.join(args.out, "all_predictions.csv"), all_rows,
                         cols=common.COLS + ["PMID"])
        json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
        print(f"\n{len(all_rows)} rows over {len(pmids)} papers -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
