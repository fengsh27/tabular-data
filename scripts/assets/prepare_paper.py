#!/usr/bin/env python3
"""Convert a raw PK/PE paper (HTML **or** JATS/PMC XML) into the canonical
curation input layout.

For each paper it writes an output directory ``<out>/<pmid>/`` containing:

  paper_text.md   -- title (H1) + article body as Markdown; references stripped;
                     every data table replaced by a ``[Table N]`` marker.
  abstract.md     -- abstract as Markdown.
  table_<n>.md    -- table <n> caption, footnotes, AND the table itself as a
                     Markdown table (under a ``**Table:**`` heading).
  table_<n>.html  -- table <n> as a <section> (caption + table + footnotes).
  manifest.json   -- machine-readable index: title, table count, per-table row /
                     column counts, and the marker <-> file mapping the router
                     relies on to splice tables back inline.

``table_<n>.md`` is the readable form every downstream skill should use for
routing and for reading a table's content: it is plain text, and it is 4-40x
smaller than the equivalent ``table_<n>.html``. The ``.html`` remains the source
of truth for the curation skills, which re-convert it themselves.

Tables are numbered by order of appearance. This is the front door of the
curation-skills suite: run it first, then point the curation / routing skills at
the produced ``<pmid>/`` directory.

Input formats (auto-detected by file extension + root element sniff):
  * **HTML** -- PMC / Wiley / Elsevier, best-effort structural selectors.
  * **JATS/PMC XML** -- the NLM/JATS ``<article>`` schema (``<front>`` meta,
    ``<body>`` sections, ``<table-wrap>`` tables, ``<ref-list>`` references).

Both formats produce the byte-compatible output layout above, so the downstream
curation skills do not care which one a paper came from.

Dependencies: ``beautifulsoup4`` for the HTML path, and for the Markdown table
conversion on either path (it is done by the bundled ``html_to_markdown_table``,
the same converter the curation skills use, so the Markdown matches theirs). The
XML path parses with the Python-3 stdlib (``xml.etree.ElementTree``); without
bs4 it still produces every output except the ``**Table:**`` block.

Usage:
    python prepare_paper.py paper.html --out ./.paper_assets
    python prepare_paper.py paper.xml  --out ./.paper_assets
    python prepare_paper.py data/papers --out ./.paper_assets   # dir of .html/.xml
    python prepare_paper.py data/papers --dry-run
"""
import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

# bs4 is needed for the HTML path and for the Markdown table conversion; import
# lazily so the XML path still parses without it.
try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - only hit when bs4 missing and HTML used
    BeautifulSoup = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONVERTER = None  # cached callable, or False once we know it is unavailable


def _converter():
    """Return the shared html->markdown table converter, or None.

    Vendored beside this script (html_to_markdown_table.py) rather than
    reimplemented, so a prepared table_<n>.md matches what the curation skills
    produce from the same table_<n>.html.
    """
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = False
        if BeautifulSoup is not None:  # the module exits if bs4 is missing
            if _HERE not in sys.path:
                sys.path.insert(0, _HERE)
            try:
                from html_to_markdown_table import single_html_table_to_markdown
                _CONVERTER = single_html_table_to_markdown
            except Exception as exc:  # pragma: no cover - missing/broken sibling
                sys.stderr.write("warning: table markdown unavailable (%s)\n" % exc)
    return _CONVERTER or None


def table_to_markdown(html):
    """Convert one table_<n>.html <section> to (markdown, n_rows, n_cols).

    Returns ("", 0, 0) when the converter is unavailable or finds no table, so a
    missing bs4 degrades table_<n>.md rather than failing the whole run.
    """
    convert = _converter()
    if not convert or not html:
        return "", 0, 0
    # Some publishers ship the same table twice inside one wrapper (e.g. an
    # inline `table-overflow` copy plus a `table-modal` popup copy). Hand the
    # converter the first <table> only, so it never has to guess.
    first = BeautifulSoup(html, "html.parser").find("table")
    if first is None:
        return "", 0, 0
    try:
        md = (convert(str(first)) or "").strip()
    except Exception as exc:  # pragma: no cover - malformed table markup
        sys.stderr.write("warning: table markdown failed (%s)\n" % exc)
        return "", 0, 0
    if not md:
        return "", 0, 0
    lines = md.split("\n")
    # line 0 is the (stacked) header, line 1 the separator, the rest are data
    n_cols = len([c for c in lines[0].split("|")[1:-1]]) if lines else 0
    n_rows = max(0, len(lines) - 2)
    return md, n_rows, n_cols


# ===========================================================================
# Shared helpers
# ===========================================================================
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


COVERAGE_MIN = 0.6


def _warn_low_coverage(path, body_md, raw_text):
    """Warn when the extracted Markdown holds much less prose than the source.

    Both serializers only emit text for tags they recognise, so an unfamiliar
    markup pattern silently yields a short paper_text.md while every other
    signal still reports success. Comparing against the container's own text
    turns that into a visible failure.
    """
    raw = len(_ws(raw_text or ""))
    got = len(body_md or "")
    if raw >= 1000 and got < COVERAGE_MIN * raw:
        sys.stderr.write(
            "warning: %s: extracted %d of ~%d body characters (%.0f%%); "
            "paper_text.md is probably missing prose\n"
            % (os.path.basename(path), got, raw, 100.0 * got / raw)
        )


def _collapse(text):
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _ws(s):
    return re.sub(r"\s+", " ", s).strip()


# ===========================================================================
# HTML path (BeautifulSoup) -- unchanged behaviour from the original script
# ===========================================================================
# Most specific first: a precise template match always beats a general one, and
# the generic `section.body` is the last selector before the heuristic fallback.
BODY_SELECTORS = [
    "section.body.main-article-body",       # PMC
    "div.article__body",                    # Wiley
    "section.article-section__content",     # Wiley (alt)
    "div.Body",                             # Elsevier (ScienceDirect, older)
    "div.xocs-content__article-container",  # Elsevier (ScienceDirect, current)
    "article.xocs-content__article",        # Elsevier (ScienceDirect, current)
    "div.article-body",                     # Atypon / Literatum
    "section#bodymatter",                   # Atypon / Literatum
    "div.c-article-body",                   # Springer Nature
    "section.body",
]
# containers that are page furniture, never the article body
CHROME_RE = re.compile(
    r"(nav|header|footer|aside|sidebar|masthead|banner|cookie|advert|menu|toolbar)", re.I
)
# tags that mark structured content; a container holding none of them is a leaf
# whose own text is a paragraph in its own right
BLOCK_TAGS = list(HEADINGS) + [
    "p", "blockquote", "ul", "ol", "table", "div", "section", "article", "figure",
]
ABSTRACT_SELECTORS = [
    "section.abstract", "div.abstract", "section#abstract1",
    "[class*=abstract]", "[id*=abstract]", "[id*=Abs]",
]
CAPTION_SELECTORS = (
    "[class*=caption]", "header.article-table-caption",
    "h2.obj_head", "h3.obj_head", ".label", ".captions",
)
FOOTNOTE_RE = re.compile(r"(foot|tw-foot|tblwrap-foot|table-footnotes|\bfn\b|legend)", re.I)
REF_RE = re.compile(r"(ref-list|references|reference-list|bibliograph|bibl)", re.I)


def _html_guess_body(soup):
    """Last resort when no BODY_SELECTOR matches: find the article body by shape.

    Publisher templates change and new ones appear; failing to find a body means
    no paper_text.md at all, which silently starves every full-text pipeline. So
    when the selector list misses, pick the **deepest** container that still
    holds essentially all of the page's prose -- that drills past the stack of
    layout wrappers (which all carry the same text) to the innermost element
    that actually contains the article.

    Returns (element, identifier) or (None, None).
    """
    cands = []
    for el in soup.find_all(["article", "main", "section", "div"]):
        ident = " ".join(
            filter(None, [el.get("id") or "", " ".join(el.get("class") or [])])
        )
        if CHROME_RE.search(ident):
            continue
        if el.find_parent(["nav", "header", "footer", "aside"]) is not None:
            continue
        n = len(el.get_text(" ", strip=True))
        if n >= 1000:
            cands.append((n, len(list(el.parents)), ident, el))
    if not cands:
        return None, None
    top = max(n for n, _, _, _ in cands)
    keep = [c for c in cands if c[0] >= 0.6 * top]
    keep.sort(key=lambda c: c[1])  # by depth
    _, _, ident, el = keep[-1]
    return el, ("<%s %s>" % (el.name, ident)).strip()


def _html_find_body(soup):
    """Return (body_element, how_it_was_found). how is None when nothing matched."""
    for sel in BODY_SELECTORS:
        el = soup.select_one(sel)
        if el:
            return el, sel
    el, ident = _html_guess_body(soup)
    if el is not None:
        return el, "heuristic %s" % ident
    return None, None


def _html_find_title(soup):
    for sel in ["h1.content-title", "h1.article-title", "h1"]:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(" ", strip=True)
            t = re.sub(r"^(Research paper|Original article|Article)\s+", "", t, flags=re.I)
            if t:
                return t
    return None


def _html_find(soup, selectors):
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el
    return None


def _html_find_wrap(table):
    for anc in table.parents:
        if anc.name not in ("section", "div", "figure"):
            continue
        id_ = anc.get("id") or ""
        cls = " ".join(anc.get("class") or [])
        if "article-table-content-wrapper" in cls:
            continue
        if (
            re.search(r"(?i)(^|[-_])t(bl)?[-_]?\d", id_)
            or "tbl" in id_.lower()
            or re.search(r"\b(tw|table-wrap|tables|article-table-content)\b", cls)
        ):
            return anc
    return None


def _html_strip_references(el):
    targets = []
    for node in el.find_all(["section", "div", "ol", "ul"]):
        attrs = (node.get("id") or "") + " " + " ".join(node.get("class") or [])
        if REF_RE.search(attrs):
            targets.append(node)
    for node in targets:
        if getattr(node, "decomposed", False) or node.parent is None:
            continue
        node.decompose()


def _html_caption(wrap):
    for sel in CAPTION_SELECTORS:
        el = wrap.select_one(sel)
        if el and el.get_text(strip=True):
            return _ws(el.get_text(" ", strip=True))
    return None


def _html_footnotes(wrap):
    notes, seen = [], set()
    for fn in wrap.find_all(class_=FOOTNOTE_RE):
        txt = _ws(fn.get_text(" ", strip=True))
        if txt and txt not in seen and not any(txt in s for s in seen):
            seen.add(txt)
            notes.append(txt)
    return [n for n in notes if not any(n != m and n in m for m in notes)]


def _html_block_to_md(root):
    out = []

    def emit(node):
        name = getattr(node, "name", None)
        if name is None:
            return
        if name in HEADINGS:
            txt = _ws(node.get_text(" ", strip=True))
            if txt:
                out.append("#" * HEADINGS[name] + " " + txt)
            return
        if name in ("p", "blockquote"):
            txt = _ws(node.get_text(" ", strip=True))
            if txt:
                out.append(txt)
            return
        if name in ("ul", "ol"):
            for li in node.find_all("li", recursive=False):
                txt = _ws(li.get_text(" ", strip=True))
                if txt:
                    out.append("- " + txt)
            return
        if name == "table":
            return
        # A container with no structured children is itself a paragraph. Many
        # publishers mark paragraphs up as <div>/<span> with only inline
        # children (sup, em, a); recursing past those drops the text entirely,
        # since bare text nodes are skipped above.
        if node.find(BLOCK_TAGS) is None:
            txt = _ws(node.get_text(" ", strip=True))
            if txt:
                out.append(txt)
            return
        for child in node.children:
            if getattr(child, "name", None):
                emit(child)

    for child in root.children:
        if getattr(child, "name", None):
            emit(child)
    return _collapse("\n\n".join(out))


def parse_html(path):
    if BeautifulSoup is None:
        sys.exit("beautifulsoup4 is required for HTML input: pip install beautifulsoup4")
    with open(path, encoding="utf-8") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")

    title = _html_find_title(soup)
    abstract_el = _html_find(soup, ABSTRACT_SELECTORS)
    body_el, body_src = _html_find_body(soup)

    wraps = []
    for t in soup.find_all("table"):
        w = _html_find_wrap(t)
        if w is not None and w not in wraps:
            wraps.append(w)

    abstract_md = _html_block_to_md(abstract_el) if abstract_el is not None else None

    body_md = None
    if body_el is not None:
        body_copy = BeautifulSoup(str(body_el), "html.parser")
        _html_strip_references(body_copy)
        # Number the markers from the SAME set of tables that becomes
        # table_<n>.md -- i.e. tables that have a caption/footnote wrapper.
        # Publishers also use bare <table> elements for layout (nav bars, figure
        # holders); those have no wrapper, produce no file, and must not consume
        # a number, or every marker after one of them points at the wrong file.
        placed = []
        for t in body_copy.find_all("table"):
            w = _html_find_wrap(t)
            if w is None:
                continue                 # layout table: no file, so no marker
            if w.parent is None or w in placed:
                continue                 # same table twice (inline + modal copy)
            placed.append(w)
            marker = body_copy.new_tag("p")
            marker.string = f"[Table {len(placed)}]"
            w.replace_with(marker)
        body_md = _html_block_to_md(body_copy)
        if len(placed) != len(wraps):
            # Tables that never appear in the body still need to be reachable by
            # anything that splices at markers.
            sys.stderr.write(
                "warning: %s: %d table marker(s) in the body but %d table file(s); "
                "appending the rest in a trailing 'Tables' section\n"
                % (os.path.basename(path), len(placed), len(wraps))
            )
            extra = [f"[Table {n}]" for n in range(len(placed) + 1, len(wraps) + 1)]
            if extra:
                body_md = body_md.rstrip() + "\n\n## Tables\n\n" + "\n\n".join(extra)
        _warn_low_coverage(path, body_md, body_copy.get_text(" ", strip=True))

    tables = []
    for n, wrap in enumerate(wraps, 1):
        wrap_copy = BeautifulSoup(str(wrap), "html.parser")
        root = wrap_copy.find(True)
        if root is not None and root.name != "section":
            root.name = "section"
        tables.append({
            "caption": _html_caption(wrap),
            "footnotes": _html_footnotes(wrap),
            "html": wrap_copy.prettify(),
        })

    return {
        "title": title,
        "abstract_md": abstract_md,
        "body_md": body_md,
        "has_body": body_el is not None,
        "body_src": body_src,
        "tables": tables,
    }


# ===========================================================================
# JATS / PMC XML path (stdlib ElementTree)
# ===========================================================================
def _localname(tag):
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xml_text(el):
    return _ws("".join(el.itertext())) if el is not None else ""


def _xml_block_to_md(container, depth, markers):
    """Serialize a JATS block (<body>/<abstract>/<sec>) to Markdown.

    headings  <- <sec><title> (level = depth+2, capped at h6)
    paragraphs<- <p>
    list items<- <list><list-item>
    tables    <- replaced by markers[id(table-wrap)] when present, else dropped
    references<- <ref-list> dropped
    """
    out = []

    def walk_children(node, depth, skip=None):
        for child in list(node):
            if child is skip:
                continue
            process(child, depth)

    def process(child, depth):
        tag = _localname(child.tag)
        if tag == "sec":
            title_el = child.find("title")
            if title_el is not None:
                txt = _xml_text(title_el)
                if txt:
                    out.append("#" * min(depth + 2, 6) + " " + txt)
            # recurse the section's children, but not the title we just emitted
            walk_children(child, depth + 1, skip=title_el)
        elif tag == "title":
            txt = _xml_text(child)
            if txt:
                out.append("#" * min(depth + 2, 6) + " " + txt)
        elif tag == "p":
            txt = _xml_text(child)
            if txt:
                out.append(txt)
        elif tag == "list":
            for li in child.findall("list-item"):
                txt = _xml_text(li)
                if txt:
                    out.append("- " + txt)
        elif tag == "table-wrap":
            mk = markers.get(id(child))
            if mk:
                out.append(mk)
        elif tag in ("ref-list", "fn-group", "table-wrap-foot"):
            return
        else:
            walk_children(child, depth)

    walk_children(container, depth)
    return _collapse("\n\n".join(out))


def _xml_table_footnotes(table_wrap):
    foot = table_wrap.find("table-wrap-foot")
    if foot is None:
        return []
    notes = []
    fns = foot.findall(".//fn")
    if fns:
        for fn in fns:
            txt = _xml_text(fn)
            if txt:
                notes.append(txt)
    else:
        txt = _xml_text(foot)
        if txt:
            notes.append(txt)
    return notes


def _xml_table_html(table_wrap):
    """Serialize a <table-wrap> as a <section> (caption + <table> + foot)."""
    clone = ET.fromstring(ET.tostring(table_wrap, encoding="unicode"))
    clone.tag = "section"
    # drop xml namespaces from tags for clean HTML-ish output
    for e in clone.iter():
        e.tag = _localname(e.tag)
        e.attrib = {_localname(k): v for k, v in e.attrib.items()}
    return ET.tostring(clone, encoding="unicode")


def parse_xml(path):
    root = ET.parse(path).getroot()

    title_el = root.find(".//front//article-title")
    title = _xml_text(title_el) or None

    abstract_el = root.find(".//front//abstract")
    abstract_md = _xml_block_to_md(abstract_el, 0, {}) if abstract_el is not None else None

    body_el = root.find(".//body")

    # tables in document order (whole article, to catch floats-group too)
    table_wraps = root.findall(".//table-wrap")
    markers = {id(tw): f"[Table {n}]" for n, tw in enumerate(table_wraps, 1)}

    body_md = _xml_block_to_md(body_el, 0, markers) if body_el is not None else None

    # JATS articles routinely keep tables outside <body> -- in a <floats-group>,
    # or as siblings of the body. Those <table-wrap>s get a number and a file,
    # but _xml_block_to_md never walks them, so their marker never reaches
    # paper_text.md and the table is invisible to anything that splices at
    # markers. Append the unplaced ones, as the legacy XmlTableParser did with
    # its trailing "Tables" section.
    if body_md is not None and table_wraps:
        orphans = [markers[id(tw)] for tw in table_wraps
                   if markers[id(tw)] not in body_md]
        if orphans:
            body_md = body_md.rstrip() + "\n\n## Tables\n\n" + "\n\n".join(orphans)

    if body_md is not None:
        # same guard as the HTML path: compare against the body's own text, with
        # tables and references excluded since those are deliberately dropped
        raw = ET.fromstring(ET.tostring(body_el, encoding="unicode"))
        for parent in raw.iter():
            for child in list(parent):
                if _localname(child.tag) in ("table-wrap", "ref-list", "fn-group"):
                    parent.remove(child)
        _warn_low_coverage(path, body_md, "".join(raw.itertext()))

    tables = []
    for tw in table_wraps:
        label = _xml_text(tw.find("label"))
        cap = _xml_text(tw.find("caption"))
        caption = ": ".join(p for p in (label, cap) if p) or None
        tables.append({
            "caption": caption,
            "footnotes": _xml_table_footnotes(tw),
            "html": _xml_table_html(tw),
        })

    return {
        "title": title,
        "abstract_md": abstract_md,
        "body_md": body_md,
        "has_body": body_el is not None,
        "tables": tables,
    }


# ===========================================================================
# Format detection + shared output writer
# ===========================================================================
def detect_format(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xml":
        return "xml"
    if ext in (".html", ".htm"):
        return "html"
    with open(path, encoding="utf-8", errors="ignore") as fh:
        head = fh.read(4096).lower()
    if "<!doctype html" in head or "<html" in head:
        return "html"
    if re.search(r"<article[\s>]", head) and "<?xml" in head:
        return "xml"
    return "html"


def write_outputs(parsed, pmid, out_root):
    out_dir = os.path.join(out_root, pmid)
    os.makedirs(out_dir, exist_ok=True)

    if parsed["abstract_md"]:
        with open(os.path.join(out_dir, "abstract.md"), "w", encoding="utf-8") as f:
            f.write(parsed["abstract_md"] + "\n")

    if parsed["body_md"] is not None:
        header = f"# {parsed['title']}\n\n" if parsed["title"] else ""
        with open(os.path.join(out_dir, "paper_text.md"), "w", encoding="utf-8") as f:
            f.write(header + parsed["body_md"] + "\n")

    tables_manifest = []
    for n, tbl in enumerate(parsed["tables"], 1):
        md_table, n_rows, n_cols = table_to_markdown(tbl["html"])
        lines = [f"# Table {n}", ""]
        if tbl["caption"]:
            lines += [tbl["caption"], ""]
        if tbl["footnotes"]:
            lines += ["**Footnotes:**", ""]
            lines += [f"- {fn}" for fn in tbl["footnotes"]]
            lines.append("")
        if md_table:
            lines += ["**Table:**", "", md_table, ""]
        with open(os.path.join(out_dir, f"table_{n}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        with open(os.path.join(out_dir, f"table_{n}.html"), "w", encoding="utf-8") as f:
            f.write(tbl["html"])
        tables_manifest.append({
            "n": n, "marker": f"[Table {n}]",
            "md": f"table_{n}.md", "html": f"table_{n}.html",
            "n_rows": n_rows, "n_cols": n_cols,
        })

    manifest = {
        "pmid": pmid,
        "title": parsed["title"],
        "n_tables": len(parsed["tables"]),
        "has_abstract": bool(parsed["abstract_md"]),
        "has_body": parsed["has_body"],
        "tables": tables_manifest,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def process_paper(path, out_root, dry_run=False):
    pmid = os.path.splitext(os.path.basename(path))[0]
    fmt = detect_format(path)
    parsed = parse_xml(path) if fmt == "xml" else parse_html(path)
    report = {
        "pmid": pmid,
        "format": fmt,
        "title": parsed["title"],
        "abstract": bool(parsed["abstract_md"]),
        "body": parsed["has_body"],
        "body_src": parsed.get("body_src"),
        "n_tables": len(parsed["tables"]),
    }
    if not dry_run:
        write_outputs(parsed, pmid, out_root)
    return report


def iter_inputs(arg):
    if os.path.isdir(arg):
        paths = sorted(
            glob.glob(os.path.join(arg, "*.html"))
            + glob.glob(os.path.join(arg, "*.htm"))
            + glob.glob(os.path.join(arg, "*.xml"))
        )
        if not paths:
            sys.exit(f"no .html/.xml files under {arg}")
        return paths
    if os.path.isfile(arg):
        return [arg]
    sys.exit(f"input not found: {arg}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="a paper .html/.xml file, or a directory of them")
    ap.add_argument("--out", default="./.paper_assets",
                    help="output root; each paper writes to <out>/<pmid>/ (default ./.paper_assets)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    for p in iter_inputs(args.input):
        r = process_paper(p, args.out, dry_run=args.dry_run)
        flag = "" if (r["title"] and r["abstract"] and r["body"]) else "  <-- CHECK"
        # surface HOW the body was found: a heuristic hit means this publisher
        # has no selector yet, which is worth adding before the template drifts
        src = r.get("body_src") or ""
        via = f" via {src}" if src.startswith("heuristic") else ""
        print(
            f"{r['pmid']} [{r['format']}]: tables={r['n_tables']} "
            f"title={'Y' if r['title'] else 'N'} "
            f"abstract={'Y' if r['abstract'] else 'N'} "
            f"body={'Y' if r['body'] else 'N'}{via}{flag}"
        )


if __name__ == "__main__":
    main()
