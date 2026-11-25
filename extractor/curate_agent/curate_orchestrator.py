"""CurateAgent cascade orchestrator, v2: per-unit (per table, or per paper for
the full-text pipelines).

    route      PKPEIdentificationStep -> PKPEDesignStep      (2 LLM calls; skipped when
               the caller fixes `schemas=`)
    units      build_units(): the schema's unit of work - see schema_specs.UNIT_KINDS
                 table         one selected table per unit (1 selector call per schema):
                               pk_summary, pk_individual, pe_study_outcome
                 text          the paper's full text, one unit:
                               pk_drug_*, pk_specimen_*, pe_study_info
                 table_or_text selected demographic tables if any, else the full text:
                               pk_population_*
    per unit   Arm A (arms.run_cheap_arm_table, 1 call)
               ValidityChecker Tier 1+2+3 against THIS unit's source only
               escalate? -> Arm B (arms.run_expensive_arm_table, execution only)
                            ValidityChecker on Arm B (Tier 1 rate excluded)
                            ComparisonJudge (Tier 4) - always, when Arm B ran
    combine    per-unit winners, in source order -> <work_dir>/<schema>/combined.csv

v1 (git history, commit efd7393) was paper-level: Arm A on every table, one
Tier 1-3 check on the whole combined CSV against all source tables at once,
one bad table re-curating the entire paper with multi-agent mode, and the
judge only on an exact error_rate tie. Why per-table: escalation cost scales
with the failing tables instead of all of them (multi-agent is 10-14 calls per
table), Tier 3 reads one table against its own source instead of a whole-paper
prompt (3m51s and a "32 missing rows" misjudgment on PMID 16143486), and the
judge rules whenever Arm B ran - on 16143486 the lower error_rate picked the
worse arm and the judge never got asked.

Scope: `arms.SUPPORTED_SCHEMAS` - all ten pipelines (each has a simple prompt
under prompts/simple_prompts/). Anything else the route step selects is
reported under `not_handled`, not curated. The per-unit records keep the
`table_index` / `caption` keys of the per-table design (a text unit has
`table_index` 0 and `unit_kind` "text").
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from io import StringIO

import pandas as pd
from TabFuncFlow.utils.table_utils import dataframe_to_markdown

from extractor.agents.agent_factory import get_agent_llm
from extractor.agents.agent_utils import extract_pmid_info_to_db
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState, PaperTypeEnum
from extractor.agents.pk_pe_agents.pk_pe_design_step import PKPEDesignStep
from extractor.agents.pk_pe_agents.pk_pe_identification_step import PKPEIdentificationStep
from extractor.constants import PipelineTypeEnum
from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.table_utils import (
    format_source_tables,
    select_pe_tables,
    select_pk_demographic_tables,
    select_pk_individual_tables,
    select_pk_summary_tables,
)
from extractor.utils import convert_html_to_text_no_table, convert_sections_to_full_text, remove_references
from extractor.curate_agent import arms, comparison_judge
from extractor.curate_agent import validity_checker as vc
from schema_specs import UNIT_KINDS, UNIT_TABLE, UNIT_TABLE_OR_TEXT, UNIT_TEXT  # via arms' sys.path entry

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SUPPORTED_SCHEMAS = set(arms.SUPPORTED_SCHEMAS)


_TUPLE_HEADER_RE = re.compile(r"^\(\s*(['\"]).*\1\s*,?\s*\)$", re.DOTALL)


def _flatten_header(col) -> str:
    """A multi-level HTML header as one label.

    The HTML extractor builds a MultiIndex for multi-row headers; the
    pmid_info.db JSON round-trip (pmid_db.select_pmid_info / _as_dataframes)
    turns each level tuple into its string repr, so a source table rendered
    straight from the DB reads `('Unnamed: 0_level_0', 'Patient') |
    ('Venlafaxine', 'Cmax1(µg l−1)') | ...`. The simple-mode arm's historical
    runs never saw that (they read the skill converter's flattened markdown),
    and on PMID 11849190 the tuple headers made it put the M/PAUC values in
    `Time value` with an empty `Parameter type` on 12 of 48 rows. Levels are
    joined with " - ", pandas' "Unnamed: N_level_M" placeholders dropped, and
    a level repeated under itself (a colspan group) kept once.
    """
    parts = None
    if isinstance(col, tuple):
        parts = [str(p) for p in col]
    elif isinstance(col, str) and _TUPLE_HEADER_RE.match(col.strip()):
        try:
            parsed = ast.literal_eval(col.strip())
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, tuple):
            parts = [str(p) for p in parsed]
    if parts is None:
        return str(col)
    kept = [p.strip() for p in parts if p.strip() and not p.strip().startswith("Unnamed:")]
    dedup = [p for i, p in enumerate(kept) if i == 0 or p != kept[i - 1]]
    return " - ".join(dedup)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_flatten_header(c) for c in df.columns]
    return df


def write_table_markdown(tables, dest_dir) -> list[str]:
    """One markdown file per source table: `<dest_dir>/table_{i}/00_markdown_table.md`
    (1-based), caption and footnote first - the layout `run_simple_prompt.py`'s
    `find_tables()` expects, and what Tier 2/3 read as the table's source.
    Multi-level headers are flattened (see _flatten_header)."""
    paths = []
    for i, table in enumerate(tables or [], start=1):
        tdir = os.path.join(dest_dir, f"table_{i}")
        os.makedirs(tdir, exist_ok=True)
        caption = table.get("caption") or ""
        footnote = table.get("footnote") or ""
        md = dataframe_to_markdown(flatten_columns(table["table"]))
        path = os.path.join(tdir, "00_markdown_table.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"{caption}\n{footnote}\n\n{md}\n")
        paths.append(path)
    return paths


def route(pmid, paper_title, paper_abstract, full_text, tables, llm=None) -> dict:
    """Which pipelines apply to this paper: the multi-agent manager's own
    identification + design steps (`PKPEManager._identification_and_design_step`),
    2 LLM calls. Returns {"paper_type": str, "schemas": [PipelineTypeEnum values]}."""
    llm = llm or get_agent_llm()
    state = PKPECurationWorkflowState(
        pmid=pmid, paper_title=paper_title, paper_abstract=paper_abstract,
        full_text=full_text, step_output_callback=None,
        source_tables=format_source_tables(tables),
    )
    state = PKPEIdentificationStep(llm=llm).execute(state)
    paper_type = state["paper_type"]
    if paper_type == PaperTypeEnum.Neither:
        return {"paper_type": paper_type.value, "schemas": []}
    state = PKPEDesignStep(llm=llm).execute(state)

    schemas = []
    for tool in state.get("pipeline_tools") or []:
        try:
            schemas.append(PipelineTypeEnum(tool).value)
        except ValueError:
            logger.warning("[%s] design step selected unknown pipeline tool %r", pmid, tool)
    return {"paper_type": paper_type.value, "schemas": schemas}


def _as_dataframes(tables) -> list[dict]:
    """extract_pmid_info_to_db()'s first-ever call for a pmid hands back tables
    whose "table" is the JSON string insert_pmid_info() serialized in place
    (extractor/database/pmid_db.py); the cached path returns DataFrames. Make
    both look like the cached path."""
    for table in tables:
        if isinstance(table.get("table"), str):
            table["table"] = pd.read_json(StringIO(table["table"]))
    return tables


def _positions_of(tables, selected) -> list[int]:
    """0-based positions in `tables` of the dicts the selector returned (it hands
    back the same objects, so identity is enough; fall back to caption+shape)."""
    positions = []
    for sel in selected:
        pos = next((i for i, t in enumerate(tables) if t is sel), None)
        if pos is None:
            pos = next(
                (i for i, t in enumerate(tables)
                 if t.get("caption") == sel.get("caption")
                 and getattr(t.get("table"), "shape", None) == getattr(sel.get("table"), "shape", None)),
                None,
            )
        if pos is not None and pos not in positions:
            positions.append(pos)
    return sorted(positions)


_SELECTORS = {
    "pk_summary": select_pk_summary_tables,
    "pk_individual": select_pk_individual_tables,
    "pe_study_outcome": select_pe_tables,
    "pk_population_summary": select_pk_demographic_tables,
    "pk_population_individual": select_pk_demographic_tables,
}


def select_tables(tables, schema, llm=None) -> list[int]:
    """0-based positions of the tables relevant to `schema` (1 LLM call).

    pk_summary uses the selector the multi-agent tools use
    (`select_pk_summary_tables`). pk_individual has its own,
    `select_pk_individual_tables`: the summary selector's inclusion rules
    ("any PK data") kept a means-only table (PMID 18426260 table 2) and a
    literature table of other papers' cases (PMID 34746508 table 3) for
    individual-level curation - see the Tier 4 audit in
    docs/cascade_mode_results.md. The production multi-agent
    `PKIndividualTablesCurationTool` still uses the summary selector; switching
    it is a separate decision because it moves that benchmark's baselines.
    pe_study_outcome and pk_population_* use the selectors their production
    tools use (`select_pe_tables`, `select_pk_demographic_tables`). Full-text
    schemas have no selector: `build_units` never calls this for them.
    """
    selector = _SELECTORS.get(schema, select_pk_summary_tables)
    selected, _, _, _ = selector(tables, llm or get_agent_llm())
    return _positions_of(tables, selected or [])


def article_text_of(title, abstract, sections, full_text=None) -> str:
    """The text the full-text pipelines curate from, built exactly as
    `FullTextCurationTool._run` / `PKPopulationSummaryCurationTool._run` do:
    the paper's sections ("<section>\n<content>" each) with HTML and tables
    stripped and the reference list cut off; title + abstract when the paper
    has no sections."""
    if sections:
        return convert_sections_to_full_text(sections)
    if full_text:
        return full_text
    text = f"{title or ''}\n{abstract or ''}"
    return remove_references(convert_html_to_text_no_table(text))


def write_full_text(article_text: str, dest_dir: str) -> str:
    """`<dest_dir>/full_text.md` - the one source file of a text unit, what
    Arm A's prompt, Tier 2 and Tier 3 read."""
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, "full_text.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(article_text)
    return path


def build_units(schema, tables, source_paths, article_text, work_dir, select_tables_fn) -> tuple[list[dict], float]:
    """The schema's units of work (see the module docstring), in source order.

    Each unit: {"kind": "table" | "text", "table_index": 1-based or 0,
    "caption", "source_path", "payload"} where `payload` is what Arm B gets:
    the pmid_db table dict for a table unit, {"text": ..., "table": None,
    "caption": "full text"} for a text unit. Returns (units, selector seconds).
    """
    kind = UNIT_KINDS.get(schema, UNIT_TABLE)
    select_seconds = 0.0
    positions = []
    if kind in (UNIT_TABLE, UNIT_TABLE_OR_TEXT) and tables:
        t0 = time.time()
        positions = select_tables_fn(tables=tables, schema=schema)
        select_seconds = time.time() - t0
    units = [{
        "kind": "table", "table_index": pos + 1,
        "caption": (tables[pos].get("caption") or "")[:200],
        "source_path": source_paths[pos], "payload": tables[pos],
    } for pos in positions]
    if kind == UNIT_TEXT or (kind == UNIT_TABLE_OR_TEXT and not units):
        path = write_full_text(article_text, os.path.join(work_dir, "source_text"))
        units = [{
            "kind": "text", "table_index": 0, "caption": "full text",
            "source_path": path,
            "payload": {"caption": "full text", "footnote": "", "table": None, "text": article_text},
        }]
    return units, select_seconds


def _table_markdown(csv_path: str) -> str:
    return dataframe_to_markdown(pd.read_csv(csv_path))


def _rule_verdict(report_a, report_b):
    """Deterministic Tier 4 verdicts that need no LLM; None when the judge
    must decide.

    An empty candidate whose empty-table assessment says the source DOES hold
    curatable data is a plain failure; against a candidate with rows it loses.
    In the second live round the reports-only judge read that assessment text
    in Arm B's report as a point in B's favour and picked an empty Arm B over a
    12-row Arm A (PMID 22050870 table 3, paper F1 0.690 -> 0.222). Both empty
    and unjustified -> keep A (nothing to choose between). A *justified* empty
    candidate is not decided here - see the caller.
    """
    def unjustified_empty(report):
        return report["tier1"]["total_rows"] == 0 and \
            not (report["tier3"] or {}).get("empty_table_justified", False)

    a_empty, b_empty = unjustified_empty(report_a), unjustified_empty(report_b)
    if b_empty and not a_empty and report_a["tier1"]["total_rows"] > 0:
        return {"winner": "A", "used_full_tables": False, "rule": "unjustified_empty_loses",
                "explanation": "Rule: Arm B returned no rows for a table whose assessment says it "
                               "has curatable data; Arm A has rows."}
    if a_empty and not b_empty and report_b["tier1"]["total_rows"] > 0:
        return {"winner": "B", "used_full_tables": False, "rule": "unjustified_empty_loses",
                "explanation": "Rule: Arm A returned no rows for a table whose assessment says it "
                               "has curatable data; Arm B has rows."}
    if a_empty and b_empty:
        return {"winner": "A", "used_full_tables": False, "rule": "both_empty",
                "explanation": "Rule: both arms returned no rows; nothing to choose between."}
    return None


def _curate_table(pmid, schema, unit, paper_title, paper_abstract, full_text,
                  table_dir, threshold, cheap_model,
                  run_cheap_arm_fn, run_expensive_arm_fn, verify_fn, compare_fn, empty_fn,
                  arm_b_mode=arms.DEFAULT_ARM_B_MODE) -> dict:
    """The cascade on one unit (a selected table, or the paper's full text) -
    see build_units for the `unit` dict."""
    table, source_path = unit["payload"], unit["source_path"]
    domain = arms.domain_for(schema)
    rec = {
        "table_index": unit["table_index"],
        "unit_kind": unit["kind"],
        "caption": unit["caption"],
        "escalated": False, "winner": None, "csv_path": None,
        "seconds": {}, "llm_calls": {},
    }

    t0 = time.time()
    a = run_cheap_arm_fn(pmid=pmid, schema=schema, table_md_path=source_path,
                         out_dir=os.path.join(table_dir, "arm_a"), model=cheap_model)
    rec["seconds"]["arm_a"] = time.time() - t0
    rec["llm_calls"]["arm_a"] = a.get("llm_calls", 1)

    t0 = time.time()
    report_a = vc.check_validity(a["csv_path"], schema, [source_path], paper_title, paper_abstract,
                                 extra_dropped_rows=a.get("dropped_rows", 0), threshold=threshold,
                                 verify_fn=verify_fn, empty_fn=empty_fn, domain=domain)
    rec["seconds"]["check_a"] = time.time() - t0
    rec["llm_calls"]["tier3_a"] = 1 if report_a["tier3"] is not None else 0
    rec["csv_a"], rec["report_a"] = a["csv_path"], report_a
    # An empty Arm A table that the Tier 3 assessment accepted (the source
    # table has no data of this schema's kind) wins here like any passing
    # table; flag it so the report shows why a zero-row winner is right.
    rec["empty_table_justified"] = bool((report_a["tier3"] or {}).get("empty_table_justified", False))

    if not report_a["escalate"]:
        rec["winner"], rec["csv_path"] = "A", a["csv_path"]
        return rec

    rec["escalated"] = True
    t0 = time.time()
    b = run_expensive_arm_fn(pmid=pmid, schema=schema, table=table,
                             out_dir=os.path.join(table_dir, "arm_b"),
                             title=paper_title, abstract=paper_abstract, full_text=full_text,
                             mode=arm_b_mode)
    rec["seconds"]["arm_b"] = time.time() - t0
    rec["arm_b_mode"] = b.get("mode", arm_b_mode)
    if b.get("final_answer") is not None:  # the multiagent loop's verdict on its own table
        rec["arm_b_final_answer"] = b["final_answer"]
    if b.get("error"):
        rec["arm_b_error"] = b["error"]

    t0 = time.time()
    # Tier 1 was built around Arm A's output conventions and is excluded from
    # Arm B's rate - see validity_checker.check_validity's docstring.
    report_b = vc.check_validity(b["csv_path"], schema, [source_path], paper_title, paper_abstract,
                                 threshold=threshold, verify_fn=verify_fn, include_tier1_rate=False,
                                 empty_fn=empty_fn, domain=domain)
    rec["seconds"]["check_b"] = time.time() - t0
    rec["llm_calls"]["tier3_b"] = 1 if report_b["tier3"] is not None else 0
    rec["csv_b"], rec["report_b"] = b["csv_path"], report_b

    t0 = time.time()
    judge = _rule_verdict(report_a, report_b)
    if judge is None:
        # When Arm B is a *justified* empty table (the assessment says the
        # source has no data of this kind) while Arm A produced rows, the
        # question is whether A's rows are real - only the tables answer that,
        # so skip the reports-only pass.
        b_justified_empty = bool((report_b["tier3"] or {}).get("empty_table_justified", False))
        judge = compare_fn(
            paper_title=paper_title, paper_abstract=paper_abstract,
            source_tables=(format_source_tables([table]) if unit["kind"] == "table"
                           else arms.unit_source_text(table)),
            report_a=report_a, table_a=_table_markdown(a["csv_path"]),
            report_b=report_b, table_b=_table_markdown(b["csv_path"]),
            force_full_tables=b_justified_empty, domain=domain,
        )
        rec["llm_calls"]["judge"] = 2 if judge.get("used_full_tables") else 1
    else:
        rec["llm_calls"]["judge"] = 0
    rec["seconds"]["judge"] = time.time() - t0
    rec["judge"] = judge
    rec["winner"] = judge["winner"]
    rec["csv_path"] = a["csv_path"] if judge["winner"] == "A" else b["csv_path"]
    return rec


def _curate_schema(pmid, schema, tables, source_paths, paper_title, paper_abstract, full_text,
                   work_dir, threshold, cheap_model, select_tables_fn,
                   run_cheap_arm_fn, run_expensive_arm_fn, verify_fn, compare_fn, empty_fn,
                   arm_b_mode=arms.DEFAULT_ARM_B_MODE, article_text=None) -> dict:
    schema_dir = os.path.join(work_dir, schema)
    os.makedirs(schema_dir, exist_ok=True)

    t_start = time.time()
    units, select_seconds = build_units(schema, tables, source_paths, article_text or full_text or "",
                                        work_dir, select_tables_fn)

    per_table, frames = [], []
    for unit in units:
        unit_dir = f"table_{unit['table_index']}" if unit["kind"] == "table" else "full_text"
        rec = _curate_table(
            pmid, schema, unit, paper_title, paper_abstract,
            full_text, os.path.join(schema_dir, unit_dir), threshold, cheap_model,
            run_cheap_arm_fn, run_expensive_arm_fn, verify_fn, compare_fn, empty_fn,
            arm_b_mode=arm_b_mode,
        )
        per_table.append(rec)
        frames.append(pd.read_csv(rec["csv_path"]))

    columns = arms._validity.schema_columns(schema)
    combined = (pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=columns))
    combined = combined.reindex(columns=columns)
    csv_path = os.path.join(schema_dir, "combined.csv")
    combined.to_csv(csv_path, index=False)

    llm_calls = {"select": 1 if select_seconds > 0 else 0}
    for rec in per_table:
        for k, v in rec["llm_calls"].items():
            llm_calls[k] = llm_calls.get(k, 0) + v
    return {
        "csv_path": csv_path,
        "rows": len(combined),
        "unit_kind": UNIT_KINDS.get(schema, UNIT_TABLE),
        "selected_table_indexes": [u["table_index"] for u in units if u["kind"] == "table"],
        "escalated_table_indexes": [r["table_index"] for r in per_table if r["escalated"]],
        "arm_b_mode": arm_b_mode,
        "tables": per_table,
        "seconds": {"select": select_seconds, "total": time.time() - t_start},
        "llm_calls": llm_calls,
    }


def curate(pmid, *, schemas=None, html_content=None, pmid_db=None, work_dir=None,
           threshold=vc.DEFAULT_THRESHOLD, cheap_model=arms.DEFAULT_CHEAP_MODEL,
           route_fn=route, select_tables_fn=select_tables,
           run_cheap_arm_fn=arms.run_cheap_arm_table,
           run_expensive_arm_fn=arms.run_expensive_arm_table,
           verify_fn=vc.llm_verification.verify, compare_fn=comparison_judge.compare,
           empty_fn=vc.llm_verification.assess_empty_table,
           arm_b_mode=arms.DEFAULT_ARM_B_MODE) -> dict:
    """Run the per-table cascade on one paper.

    `arm_b_mode` is "pipeline" (execution only, the default) or "multiagent"
    (execution + the production per-table verify/correct loop) - see
    arms.ARM_B_MODES; it is passed to `run_expensive_arm_fn` as `mode` and
    recorded per schema and per escalated table in the report.

    `schemas=None` runs the route step; a list of schema names skips it (the
    calibration script does this so results stay comparable with a fixed
    schema). Every `*_fn` defaults to the real implementation and is
    injectable so the cascade's own logic can be tested without an LLM,
    an Ollama server or a multi-agent run. `empty_fn` is the zero-row
    assessment `check_validity` runs instead of `verify_fn` on an empty
    table (see validity_checker.check_validity).

    Returns (also written to `<work_dir>/cascade_report.json`):
        {"pmid", "route": {...} | None, "not_handled": [schema, ...],
         "schemas": {schema: {"csv_path", "rows", "unit_kind", "selected_table_indexes",
                              "escalated_table_indexes", "tables": [per-unit records],
                              "seconds", "llm_calls"}},
         "seconds": {"route", "total"}}
    A full-text schema has one unit with `table_index` 0 (`unit_kind` "text");
    `selected_table_indexes` is then empty and `escalated_table_indexes` is
    [0] when that unit escalated.
    """
    if arm_b_mode not in arms.ARM_B_MODES:
        raise ValueError(f"unknown Arm B mode {arm_b_mode!r}; expected one of {arms.ARM_B_MODES}")
    t_start = time.time()
    pmid_db = pmid_db or PMIDDB()
    work_dir = work_dir or os.path.join(REPO_ROOT, ".curate_agent_work", pmid)
    os.makedirs(work_dir, exist_ok=True)

    _, paper_title, paper_abstract, full_text, tables, sections = extract_pmid_info_to_db(
        pmid, pmid_db, html_content=html_content
    )
    tables = _as_dataframes(tables or [])
    article_text = article_text_of(paper_title, paper_abstract, sections, full_text)
    if sections:
        full_text = article_text
    source_paths = write_table_markdown(tables, os.path.join(work_dir, "source_tables"))

    result = {"pmid": pmid, "route": None, "not_handled": [], "schemas": {}, "seconds": {}}

    t0 = time.time()
    if schemas is None:
        result["route"] = route_fn(pmid=pmid, paper_title=paper_title, paper_abstract=paper_abstract,
                                   full_text=full_text, tables=tables)
        requested = result["route"]["schemas"]
    else:
        requested = list(schemas)
    result["seconds"]["route"] = time.time() - t0

    for schema in requested:
        if schema not in SUPPORTED_SCHEMAS:
            result["not_handled"].append(schema)
            continue
        result["schemas"][schema] = _curate_schema(
            pmid, schema, tables, source_paths, paper_title, paper_abstract, full_text,
            work_dir, threshold, cheap_model, select_tables_fn,
            run_cheap_arm_fn, run_expensive_arm_fn, verify_fn, compare_fn, empty_fn,
            arm_b_mode=arm_b_mode, article_text=article_text,
        )

    result["seconds"]["total"] = time.time() - t_start
    with open(os.path.join(work_dir, "cascade_report.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    return result
