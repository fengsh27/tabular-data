#!/usr/bin/env python3
"""Convert a raw PK/PE paper (HTML) into the canonical curation input layout.

For each paper HTML it writes an output directory ``<out>/<pmid>/`` containing:

  paper_text.md   -- title (H1) + article body as Markdown; references stripped;
                     every data table replaced by a ``[Table N]`` marker.
  abstract.md     -- abstract as Markdown.
  table_<n>.md    -- table <n> caption + footnotes as Markdown.
  table_<n>.html  -- table <n> as a <section> (caption + table + footnotes).
  manifest.json   -- machine-readable index: title, table count, and the
                     marker <-> file mapping the router relies on to splice
                     tables back inline.

Tables are numbered by order of appearance. This is the front door of the
curation-skills suite: run it first, then point the curation / routing skills at
the produced ``<pmid>/`` directory.

Scope: HTML only (PMC / Wiley / Elsevier best-effort selectors). JATS/PMC *XML*
is not handled here (different structure) -- that is a planned follow-up.

Dependencies: ``beautifulsoup4`` only (no markdownify) -- the body/abstract/
caption Markdown is produced by a small bs4 serializer below so the skill stays
dep-light for Ollama/OSC hosts.

Usage:
    python prepare_paper.py paper.html --out ./.paper_assets
    python prepare_paper.py data/pk-individual --out ./.paper_assets   # a dir of <pmid>.html
    python prepare_paper.py data/pk-individual --dry-run
"""
import argparse
import glob
import json
import os
import re
import sys

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Structural locators (publisher-agnostic, best-effort) -- from extract_paper_assets
# ---------------------------------------------------------------------------
BODY_SELECTORS = [
    "section.body.main-article-body",   # PMC
    "div.article__body",                # Wiley
    "div.Body",                         # Elsevier (ScienceDirect)
    "section.body",
]

ABSTRACT_SELECTORS = [
    "section.abstract",
    "div.abstract",
    "section#abstract1",
    "[class*=abstract]",
    "[id*=abstract]",
    "[id*=Abs]",
]

CAPTION_SELECTORS = (
    "[class*=caption]",
    "header.article-table-caption",
    "h2.obj_head",
    "h3.obj_head",
    ".label",
    ".captions",
)

FOOTNOTE_RE = re.compile(r"(foot|tw-foot|tblwrap-foot|table-footnotes|\bfn\b|legend)", re.I)

# id/class patterns marking a references / bibliography block to drop from body
REF_RE = re.compile(r"(ref-list|references|reference-list|bibliograph|bibl)", re.I)

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def find_title(soup):
    for sel in ["h1.content-title", "h1.article-title", "h1"]:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(" ", strip=True)
            t = re.sub(r"^(Research paper|Original article|Article)\s+", "", t, flags=re.I)
            if t:
                return t
    return None


def find_body(soup):
    for sel in BODY_SELECTORS:
        el = soup.select_one(sel)
        if el:
            return el
    return None


def find_abstract(soup):
    for sel in ABSTRACT_SELECTORS:
        el = soup.select_one(sel)
        if el:
            return el
    return None


def find_wrap(table):
    """Climb to the nearest table-wrap ancestor (caption + table + footnotes)."""
    for anc in table.parents:
        if anc.name not in ("section", "div", "figure"):
            continue
        id_ = anc.get("id") or ""
        cls = " ".join(anc.get("class") or [])
        # the inner Wiley wrapper holds only the table, not caption/footnotes
        if "article-table-content-wrapper" in cls:
            continue
        if (
            re.search(r"(?i)(^|[-_])t(bl)?[-_]?\d", id_)
            or "tbl" in id_.lower()
            or re.search(r"\b(tw|table-wrap|tables|article-table-content)\b", cls)
        ):
            return anc
    return None


def strip_references(el):
    """Remove reference/bibliography sections (heading + list) from a body copy."""
    targets = []
    for node in el.find_all(["section", "div", "ol", "ul"]):
        attrs = (node.get("id") or "") + " " + " ".join(node.get("class") or [])
        if REF_RE.search(attrs):
            targets.append(node)
    for node in targets:
        if getattr(node, "decomposed", False) or node.parent is None:
            continue  # already removed as part of an ancestor
        node.decompose()


def extract_caption(wrap):
    for sel in CAPTION_SELECTORS:
        el = wrap.select_one(sel)
        if el and el.get_text(strip=True):
            return el
    return None


def extract_footnotes(wrap):
    notes, seen = [], set()
    for fn in wrap.find_all(class_=FOOTNOTE_RE):
        txt = re.sub(r"\s+", " ", fn.get_text(" ", strip=True)).strip()
        if txt and txt not in seen:
            if not any(txt in s for s in seen):
                seen.add(txt)
                notes.append(txt)
    notes = [n for n in notes if not any(n != m and n in m for m in notes)]
    return notes


# ---------------------------------------------------------------------------
# Markdown serialization (bs4 only -- replaces markdownify)
# ---------------------------------------------------------------------------
def _inline(node):
    """Normalized inline text of a node (whitespace-collapsed)."""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def block_to_markdown(root):
    """Serialize a block container to Markdown, emitting headings, paragraphs,
    and list items. Container nodes (section/div/...) are recursed into; inline
    markup is flattened to text. Sufficient for LLM input, not pixel-perfect.
    """
    out = []

    def emit(node):
        name = getattr(node, "name", None)
        if name is None:
            return  # NavigableString handled by its parent block
        if name in HEADINGS:
            txt = _inline(node)
            if txt:
                out.append("#" * HEADINGS[name] + " " + txt)
            return
        if name in ("p", "blockquote"):
            txt = _inline(node)
            if txt:
                out.append(txt)
            return
        if name in ("ul", "ol"):
            for li in node.find_all("li", recursive=False):
                txt = _inline(li)
                if txt:
                    out.append("- " + txt)
            return
        if name == "table":
            return  # data tables are replaced by markers before this runs
        # container: recurse element children
        for child in node.children:
            if getattr(child, "name", None):
                emit(child)

    for child in root.children:
        if getattr(child, "name", None):
            emit(child)
    text = "\n\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def caption_to_markdown(caption_el):
    return _inline(caption_el)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def process_paper(path, out_root, dry_run=False):
    pmid = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")

    title = find_title(soup)
    abstract_el = find_abstract(soup)
    body_el = find_body(soup)

    # discover data tables (those with a detectable wrap) in document order
    table_wraps = []
    for t in soup.find_all("table"):
        w = find_wrap(t)
        if w is not None and w not in table_wraps:
            table_wraps.append(w)

    report = {
        "pmid": pmid,
        "title": title,
        "abstract": abstract_el is not None,
        "body": body_el is not None,
        "n_tables": len(table_wraps),
    }
    if dry_run:
        return report

    out_dir = os.path.join(out_root, pmid)
    os.makedirs(out_dir, exist_ok=True)

    # --- abstract.md ---
    if abstract_el is not None:
        abs_md = block_to_markdown(abstract_el)
        with open(os.path.join(out_dir, "abstract.md"), "w", encoding="utf-8") as f:
            f.write(abs_md + "\n")

    # --- paper_text.md (body with tables replaced by [Table N] markers) ---
    if body_el is not None:
        body_copy = BeautifulSoup(str(body_el), "html.parser")
        strip_references(body_copy)
        for i, t in enumerate(body_copy.find_all("table"), 0):
            w = find_wrap(t) or t
            marker = body_copy.new_tag("p")
            marker.string = f"[Table {i + 1}]"
            w.replace_with(marker)
        body_md = block_to_markdown(body_copy)
        header = f"# {title}\n\n" if title else ""
        with open(os.path.join(out_dir, "paper_text.md"), "w", encoding="utf-8") as f:
            f.write(header + body_md + "\n")

    # --- per-table outputs + manifest entries ---
    tables_manifest = []
    for n, wrap in enumerate(table_wraps, 1):
        caption_el = extract_caption(wrap)
        footnotes = extract_footnotes(wrap)

        # table_<n>.md : caption + footnotes
        lines = [f"# Table {n}", ""]
        if caption_el is not None:
            lines.append(caption_to_markdown(caption_el))
            lines.append("")
        if footnotes:
            lines.append("**Footnotes:**")
            lines.append("")
            for fn in footnotes:
                lines.append(f"- {fn}")
            lines.append("")
        with open(os.path.join(out_dir, f"table_{n}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")

        # table_<n>.html : the wrap as a <section>
        wrap_copy = BeautifulSoup(str(wrap), "html.parser")
        root = wrap_copy.find(True)
        if root is not None and root.name != "section":
            root.name = "section"
        with open(os.path.join(out_dir, f"table_{n}.html"), "w", encoding="utf-8") as f:
            f.write(wrap_copy.prettify())

        tables_manifest.append({
            "n": n,
            "marker": f"[Table {n}]",
            "md": f"table_{n}.md",
            "html": f"table_{n}.html",
        })

    # --- manifest.json ---
    manifest = {
        "pmid": pmid,
        "title": title,
        "n_tables": len(table_wraps),
        "has_abstract": abstract_el is not None,
        "has_body": body_el is not None,
        "tables": tables_manifest,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return report


def iter_inputs(arg):
    if os.path.isdir(arg):
        paths = sorted(glob.glob(os.path.join(arg, "*.html")))
        if not paths:
            sys.exit(f"no .html files under {arg}")
        return paths
    if os.path.isfile(arg):
        return [arg]
    sys.exit(f"input not found: {arg}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="a paper .html file, or a directory of <pmid>.html files")
    ap.add_argument("--out", default="./.paper_assets",
                    help="output root; each paper writes to <out>/<pmid>/ (default ./.paper_assets)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    for p in iter_inputs(args.input):
        r = process_paper(p, args.out, dry_run=args.dry_run)
        flag = "" if (r["title"] and r["abstract"] and r["body"]) else "  <-- CHECK"
        print(
            f"{r['pmid']}: tables={r['n_tables']} "
            f"title={'Y' if r['title'] else 'N'} "
            f"abstract={'Y' if r['abstract'] else 'N'} "
            f"body={'Y' if r['body'] else 'N'}{flag}"
        )


if __name__ == "__main__":
    main()
