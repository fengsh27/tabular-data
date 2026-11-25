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

--backend selects the model backend: "ollama" (default, a local Ollama server
via --base-url/--model), "gpt4o" (Azure gpt-4o via
extractor.request_openai.get_openai()), or "gpt54" (Azure gpt-5.4 via
extractor.request_openai.get_5_openai()) - the two Azure backends read .env
and ignore --model. All backends read the exact same prepared table set and
write the exact same manifest.json shape (rows, input_tokens, output_tokens,
elapsed_seconds per table), so results are directly comparable.

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
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

# --schema picks the column set / parser: pk_individual (default, _common.py),
# pk_summary (_common_summary.py), or one of the eight schemas described in
# schema_specs.py and parsed by _common_generic.SchemaParser. All expose the
# same parse_csv/write_csv/COLS surface (schema_registry.load_schema_module),
# so the rest of this script does not need to know which one it got.
from schema_registry import ALL_SCHEMAS, UNIT_KINDS, load_schema_module  # noqa: E402

SCHEMA_MODULES = {schema: schema for schema in ALL_SCHEMAS}  # kept for callers that list schemas

TABLE_FILES = ("00_markdown_table.md", "inputs.md")
# A full-text schema's one unit per paper: the prepared article text.
FULL_TEXT_FILES = ("full_text.md", "paper_text.md", "inputs.md")

# backend name -> (display label, extractor.request_openai factory function name)
AZURE_BACKENDS = {
    "gpt4o": ("gpt-4o (Azure)", "get_openai"),
    "gpt54": ("gpt-5.4 (Azure)", "get_5_openai"),
}


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


def call_ollama(base_url, model, prompt, temperature, num_ctx, timeout, retries=2,
                 think=None):
    """Return the parsed /api/generate body (not just "response").

    A thinking model can spend its whole generation budget inside <think>...
    and never reach an answer; when that happens Ollama's "response" comes
    back empty while "thinking" holds everything the model actually wrote.
    Callers need both fields to tell that apart from a real empty answer.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    if think is not None:
        payload["think"] = think
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
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001 - report and retry
            last = exc
            if attempt < retries:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ollama call failed after {retries + 1} tries: {last}")


def call_azure_chat(client, prompt, label, retries=2):
    """Call an Azure chat client (gpt-4o, gpt-5.4, ...), returning a body dict
    shaped like call_ollama's (same "response"/"thinking"/"done_reason" keys,
    plus prompt_eval_count/eval_count for token counts) so the rest of main()
    does not need to know which backend it is talking to."""
    from langchain_core.messages import HumanMessage

    last = None
    for attempt in range(retries + 1):
        try:
            resp = client.invoke([HumanMessage(content=prompt)])
            usage = resp.usage_metadata or {}
            return {
                "response": resp.content or "",
                "prompt_eval_count": usage.get("input_tokens"),
                "eval_count": usage.get("output_tokens"),
                "thinking": "",
                "done_reason": "stop",
            }
        except Exception as exc:  # noqa: BLE001 - report and retry
            last = exc
            if attempt < retries:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{label} call failed after {retries + 1} tries: {last}")


def render_prompt(template: str, pmid: str, table_text: str) -> str:
    return template.replace("{PMID}", pmid).replace("{TABLE}", table_text)


def curate_table(template, pmid, table_text, common, *, base_url=None, model=None,
                 temperature=0.0, num_ctx=32768, timeout=900,
                 call_fn=None, retry_without_think=True) -> dict:
    """One simple-mode pass on one table: render the prompt, call the model,
    parse the CSV. This is the unit the CurateAgent cascade runs per table
    (extractor/curate_agent/arms.py); main() runs it in a loop.

    `call_fn(prompt, **kw) -> body` overrides the model call (Azure backends,
    tests); the default is call_ollama with the given connection settings.
    A thinking model can burn its whole budget inside <think>...</think> and
    never reach an answer - "response" comes back empty while "thinking"
    holds what it wrote - so an empty response is retried once with
    think=False, unless `retry_without_think` is off (non-thinking backends).
    The same retry fires when the response parses to no rows at all (a
    header line and nothing else: qwen3.8 did this on pk_summary tables after
    thousands of thinking tokens, 2026-10-31), since that is indistinguishable
    from the empty case for the caller.

    Returns {"rows", "dropped", "raw", "thinking", "prompt", "elapsed_seconds",
    "retried_no_think", "input_tokens", "output_tokens"}; "thinking" is the
    model's reasoning text when the backend returns it (kept for diagnosis).
    """
    prompt = render_prompt(template, pmid, table_text)
    if call_fn is None:
        def call_fn(p, **kw):
            return call_ollama(base_url, model, p, temperature, num_ctx, timeout, **kw)

    t0 = time.time()
    body = call_fn(prompt)
    raw = body.get("response", "")
    thinking = body.get("thinking", "") or ""
    dropped = []
    rows = common.parse_csv(raw, pmid=pmid, dropped=dropped)
    retried = False
    if not rows and not dropped and retry_without_think:
        body = call_fn(prompt, think=False)
        raw = body.get("response", "")
        thinking = thinking or (body.get("thinking", "") or "")
        dropped = []
        rows = common.parse_csv(raw, pmid=pmid, dropped=dropped)
        retried = True
    elapsed = time.time() - t0

    return {
        "rows": rows, "dropped": dropped, "raw": raw, "thinking": thinking, "prompt": prompt,
        "elapsed_seconds": elapsed, "retried_no_think": retried,
        "input_tokens": body.get("prompt_eval_count"),
        "output_tokens": body.get("eval_count"),
    }


def find_tables(scratch: str, pmid: str, schema: str = "pk_individual"):
    """The table directories the skill selected, with the file to feed the model.

    For a full-text schema (`UNIT_KINDS[schema] == "text"`) the paper's one
    unit is `<scratch>/<pmid>/full_text.md` (or paper_text.md / inputs.md),
    reported under the name "full_text" so the output files line up with the
    cascade's (`arms.run_cheap_arm_table` writes the same shape).
    """
    root = os.path.join(scratch, pmid)
    found = []
    if UNIT_KINDS.get(schema) == "text":
        for candidate in FULL_TEXT_FILES:
            path = os.path.join(root, candidate)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return [("full_text", path, candidate)]
        print(f"  [warn] {pmid}: no full-text file ({', '.join(FULL_TEXT_FILES)}), skipped")
        return []
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
    ap.add_argument("--schema", choices=sorted(ALL_SCHEMAS), default="pk_individual",
                    help="column set / parser to use (default: pk_individual). Full-text schemas "
                         "(pk_drug_*, pk_specimen_*, pe_study_info) read <scratch>/<pmid>/full_text.md "
                         "as their one 'table'")
    ap.add_argument("--backend", choices=["ollama", *AZURE_BACKENDS], default="ollama",
                    help="model backend: local Ollama server (default), or an "
                         "Azure chat deployment (gpt4o, gpt54) via extractor.request_openai")
    ap.add_argument("--model", default=os.environ.get("SIMPLE_PROMPT_MODEL", "qwen3.8:27b"),
                    help="Ollama model tag; ignored for an Azure backend")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--pmids", default=None, help="comma list, or a file of PMIDs")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--resume", action="store_true", help="skip tables already done")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    common = load_schema_module(args.schema)
    base_url = resolve_base_url(args.base_url) if args.backend == "ollama" else None
    template = load_prompt(args.prompt)
    if "{TABLE}" not in template:
        print(f"[error] {args.prompt} has no {{TABLE}} placeholder", file=sys.stderr)
        return 2

    client = None
    if args.backend in AZURE_BACKENDS and not args.dry_run:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
        import extractor.request_openai as request_openai
        _, factory_name = AZURE_BACKENDS[args.backend]
        client = getattr(request_openai, factory_name)()

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
    model_desc = (AZURE_BACKENDS[args.backend][0] if args.backend in AZURE_BACKENDS
                  else f"{args.model} base_url={base_url}")
    print(f"backend={args.backend} model={model_desc} papers={len(pmids)} "
          f"temp={args.temperature}{' [dry-run]' if args.dry_run else ''}")

    all_rows, manifest = [], []
    for pmid in pmids:
        pdir = os.path.join(args.out, pmid)
        os.makedirs(pdir, exist_ok=True)
        tables = find_tables(args.scratch, pmid, args.schema)
        print(f"[{pmid}] {len(tables)} table(s): {', '.join(t[0] for t in tables)}")

        rows = []
        for tname, tpath, source in tables:
            raw_path = os.path.join(pdir, f"{tname}.raw.txt")
            table_text = open(tpath, encoding="utf-8").read()

            if args.dry_run:
                open(os.path.join(pdir, f"{tname}.prompt.txt"), "w",
                     encoding="utf-8").write(render_prompt(template, pmid, table_text))
                continue

            if args.resume and os.path.exists(raw_path):
                raw = open(raw_path, encoding="utf-8").read()
                dropped = []
                result = {"rows": common.parse_csv(raw, pmid=pmid, dropped=dropped),
                          "dropped": dropped, "retried_no_think": False,
                          "input_tokens": None, "output_tokens": None, "elapsed_seconds": None}
                note = "cached"
            else:
                try:
                    if args.backend in AZURE_BACKENDS:
                        label = AZURE_BACKENDS[args.backend][0]
                        result = curate_table(
                            template, pmid, table_text, common,
                            call_fn=lambda p, **kw: call_azure_chat(client, p, label),
                            retry_without_think=False,  # not a thinking model
                        )
                    else:
                        result = curate_table(
                            template, pmid, table_text, common,
                            base_url=base_url, model=args.model, temperature=args.temperature,
                            num_ctx=args.num_ctx, timeout=args.timeout,
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [error] {tname}: {exc}")
                    manifest.append({"pmid": pmid, "table": tname, "error": str(exc)})
                    continue
                open(raw_path, "w", encoding="utf-8").write(result["raw"])
                note = f"{result['elapsed_seconds']:.0f}s"
                if result["retried_no_think"]:
                    note += ", think=false retry"

            parsed, dropped = result["rows"], result["dropped"]
            rows.extend(parsed)
            lost = f", {len(dropped)} unparsable" if dropped else ""
            print(f"  {tname}: {len(parsed)} rows ({note}, from {source}){lost}")
            for bad in dropped:
                print(f"    [dropped] {len(bad)} fields: {','.join(bad)[:110]}")
            elapsed_seconds = result["elapsed_seconds"]
            manifest.append({"pmid": pmid, "table": tname, "source": source,
                             "rows": len(parsed), "dropped": len(dropped),
                             "retried_no_think": result["retried_no_think"],
                             "input_tokens": result["input_tokens"],
                             "output_tokens": result["output_tokens"],
                             "elapsed_seconds": (
                                 round(elapsed_seconds, 3) if elapsed_seconds is not None else None
                             )})

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
