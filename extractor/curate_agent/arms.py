"""The two curation arms of the CurateAgent cascade, each scoped to ONE table.

Arm A (cheap): the simple-prompt path - one model call on the table's markdown,
parsed with the schema's `parse_csv` (`scripts/simple_prompt/`). Runs
in-process via `run_simple_prompt.curate_table`, the same function its CLI
loops over, so the cascade and the standalone script produce identical output.

Arm B (expensive): the LangGraph workflow for the schema (`PKSumWorkflow` /
`PKIndWorkflow`), executed on the one table exactly the way
`PKSummaryTablesCurationTool._run` / `PKIndividualTablesCurationTool._run`
(extractor/agents/pk_pe_agents/pk_pe_agent_tools.py) do per selected table,
without table selection (the orchestrator did that). It has two modes,
named after the two PKPEManager modes they reproduce (`ARM_B_MODES`):

  "pipeline"   (default) execution only - `go_md_table` and nothing else, the
               same as PKPEManager with `enable_verification=False`.
  "multiagent" execution followed by the production per-table verify/correct
               loop (`PKPEPerTableVerifyCorrectStep._run_one_table`, i.e.
               PKPEManager's default VerifyScopeEnum.PerTable), scoped to this
               one table: up to MAX_PER_TABLE_STEP_COUNT rounds of
               verification (+ code correction while "Incorrect"), 1-11 extra
               LLM calls on AGENT_LLM.

Benchmark evidence for making "pipeline" the default: multiagent vs pipeline
mode (same workflow, with vs without the loop) differs in strict macro F1 by
+0.005 on pk-summary and -0.002 on pk-individual, for 1-25 extra LLM calls
per table; the cascade's own Tier 3 check and Tier 4 judge take over that
role. The switch exists so the two can be compared inside the cascade.

Scope: all ten pipelines (`SUPPORTED_SCHEMAS`). The unit of work differs by
schema (`schema_specs.UNIT_KINDS`): pk_summary, pk_individual and
pe_study_outcome curate one selected table per unit (the production
`go_md_table` path); pk_drug_*, pk_specimen_* and pe_study_info read the
paper's full text as one unit per paper (`FullTextCurationTool` ->
`go_full_text`); pk_population_* take each selected demographic table as a
unit, or the full text when there is none (`PKPopulationSummaryCurationTool`).
`run_cheap_arm_table` reads whatever file the orchestrator wrote for the unit
(a markdown table or the article text); `run_expensive_arm_table` hands the
unit to the schema's workflow the way its production tool does
(`WORKFLOW_INPUT`).

Both arms write a CSV on the schema's canonical columns
(`scripts/simple_prompt/validity.schema_columns`). The multi-agent pk_summary
workflow names five columns differently from the gold/simple-mode schema
(`Parameter value` vs `Value`, ...) - `normalize_to_schema` maps them, so Tier
1's by-name lookups see real values (on PMID 16143486 they used to come back
"" for 5 of 16 columns) and per-table winners from either arm concatenate
cleanly.
"""

from __future__ import annotations

import csv
import importlib
import logging
import os
import sys
import time

import pandas as pd
from TabFuncFlow.utils.table_utils import dataframe_to_markdown

from extractor.agents.agent_factory import get_agent_llm, get_pipeline_llm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SIMPLE_PROMPT_DIR = os.path.join(REPO_ROOT, "scripts", "simple_prompt")
if _SIMPLE_PROMPT_DIR not in sys.path:
    sys.path.insert(0, _SIMPLE_PROMPT_DIR)
import run_simple_prompt as _rsp  # scripts/simple_prompt/run_simple_prompt.py
import validity as _validity  # scripts/simple_prompt/validity.py
from schema_registry import ALL_SCHEMAS, UNIT_KINDS, load_schema_module  # scripts/simple_prompt/schema_registry.py

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMAS = tuple(ALL_SCHEMAS)

# What each schema's workflow entry point takes: the selected table as
# markdown (`go_md_table`) or a block of text (`go_full_text`). A table unit
# of a full-text workflow (pk_population_* on a demographic table) is passed
# as "<markdown table>\n\n<caption + footnote>", exactly as
# PKPopulationSummaryCurationTool._run does.
WORKFLOW_INPUT = {
    "pk_summary": "md_table", "pk_individual": "md_table", "pe_study_outcome": "md_table",
    "pk_drug_summary": "full_text", "pk_drug_individual": "full_text",
    "pk_specimen_summary": "full_text", "pk_specimen_individual": "full_text",
    "pk_population_summary": "full_text", "pk_population_individual": "full_text",
    "pe_study_info": "full_text",
}

DEFAULT_PROMPT_PATHS = {
    schema: os.path.join(REPO_ROOT, "prompts", "simple_prompts", f"{schema}.md")
    for schema in SUPPORTED_SCHEMAS
}

DEFAULT_CHEAP_MODEL = "qwen3.8:27b"

ARM_B_MODES = ("pipeline", "multiagent")
DEFAULT_ARM_B_MODE = "pipeline"

# The `domain` string each PKPEAgentToolTask subclass hands its verify/correct
# steps (pk_summary_task._get_domain / pk_individual_task._get_domain / ...);
# the cascade's Tier 3 and Tier 4 prompts use the same wording.
_VERIFY_DOMAINS = {
    "pk_summary": "pharmacokinetic summary",
    "pk_individual": "pharmacokinetic population and individual",
    "pk_drug_summary": "pharmacokinetic drug dosing (study-level)",
    "pk_drug_individual": "pharmacokinetic drug dosing (per individual)",
    "pk_specimen_summary": "pharmacokinetic specimen sampling (study-level)",
    "pk_specimen_individual": "pharmacokinetic specimen sampling (per individual)",
    "pk_population_summary": "pharmacokinetic study population characteristics (summary)",
    "pk_population_individual": "pharmacokinetic study population characteristics (per individual)",
    "pe_study_info": "pharmacoepidemiology study design",
    "pe_study_outcome": "pharmacoepidemiology study outcomes",
}


def domain_for(schema: str) -> str:
    return _VERIFY_DOMAINS.get(schema, "pharmacokinetics")

# PKSumWorkflow.go_md_table output -> the pk_summary gold / simple-mode columns.
# pk_individual's workflow already emits the canonical 12 columns.
PK_SUMMARY_MULTIAGENT_TO_CANONICAL = {
    "Parameter value": "Value",
    "Parameter unit": "Unit",
    "Parameter statistic": "Summary statistics",
    "Lower bound": "Lower limit",
    "Upper bound": "High limit",
}

# Workflow output column -> canonical column, per schema. The other workflows
# already rename to the canonical names in their go_full_text/go_md_table
# (e.g. "Population N" -> "Subject N", "Source text" -> "Note"); the entries
# below only cover the pre-rename spellings in case a workflow is bypassed.
_POPULATION_PRE_RENAME = {
    "Population N": "Subject N", "Source text": "Note",
    "Population characteristic": "Characteristic", "Patient characteristic": "Characteristic",
    "Characteristic sub-category": "Characteristic subcategory",
    "Unit": "Characteristic unit", "Main value": "Characteristic value",
}
MULTIAGENT_TO_CANONICAL = {
    "pk_summary": PK_SUMMARY_MULTIAGENT_TO_CANONICAL,
    "pk_drug_summary": {"Population N": "Subject N", "Source text": "Note"},
    "pk_drug_individual": {"Source text": "Note"},
    "pk_specimen_summary": {"Population N": "Subject N", "Source text": "Note"},
    "pk_specimen_individual": {"Source text": "Note"},
    "pk_population_summary": _POPULATION_PRE_RENAME,
    "pk_population_individual": _POPULATION_PRE_RENAME,
    "pe_study_info": {"Source text": "Note"},
    "pe_study_outcome": {"Outcomes": "Outcome", "Main value": "Parameter value",
                         "Statistics type": "Parameter statistic", "Main value unit": "Parameter unit"},
}


def schema_common(schema: str):
    """The schema's parse_csv/write_csv/COLS module (`_common`, `_common_summary`,
    or a `SchemaParser` for the generic schemas)."""
    return load_schema_module(schema)


def normalize_to_schema(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    """Canonical column names and order for `schema`; extra columns dropped,
    missing ones empty."""
    mapping = MULTIAGENT_TO_CANONICAL.get(schema)
    if mapping:
        df = df.rename(columns={k: v for k, v in mapping.items() if v not in df.columns})
    return df.reindex(columns=_validity.schema_columns(schema))


_PLACEHOLDER_CELLS = {"", "n/a", "na", "none", "nan", "null", "not applicable", "-"}


def drop_placeholder_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose every cell is a "nothing found" placeholder.

    The full-text workflows answer "no such data" with one row of "N/A" in
    every column (their post-processing asks the model for exactly that -
    e.g. pk_spec_ind_specimen_info_agent.post_process_specimen_info). To the
    cascade that row is an empty table: on PMID 18701886 (pk_specimen_individual,
    first pk-pe round) it reached Tier 3 as a 1-row table and was rated "All
    values match", when the empty-table assessment is the check that applies.
    """
    def is_placeholder(v):
        if v is None:
            return True
        try:
            if isinstance(v, float) and v != v:  # NaN
                return True
        except TypeError:
            pass
        return str(v).strip().lower() in _PLACEHOLDER_CELLS

    keep = [not all(is_placeholder(v) for v in row) for row in df.itertuples(index=False)]
    return df[keep].reset_index(drop=True)


def unit_source_text(unit: dict) -> str:
    """The text a unit is curated from: a text unit's `text`; a table unit's
    markdown table followed by its caption and footnote (the production
    tools' `source_table` for a demographic table)."""
    if unit.get("table") is None:
        return unit.get("text") or ""
    caption = "\n".join([unit.get("caption") or "", unit.get("footnote") or ""]).strip()
    md = dataframe_to_markdown(unit["table"])
    return f"{md}\n\n{caption}" if caption else md


def write_empty_csv(csv_path: str, schema: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(_validity.schema_columns(schema))


def run_cheap_arm_table(pmid, schema, table_md_path, out_dir, *,
                        model=DEFAULT_CHEAP_MODEL, base_url=None, temperature=0.0,
                        num_ctx=32768, timeout=900, prompt_path=None, call_fn=None) -> dict:
    """Arm A on one unit (1 LLM call; 2 if the thinking-model empty-response
    retry fires). `table_md_path` is the unit's source file - a markdown table
    or, for a full-text schema, the article text - and goes into the prompt's
    {TABLE} slot. Writes `<out_dir>/arm_a.csv` (+ the raw model reply).

    Returns {"csv_path", "rows", "dropped_rows", "seconds", "llm_calls",
    "input_tokens", "output_tokens"}. `dropped_rows` is what `parse_csv`
    could not realign - Tier 1's `extra_dropped_rows`.
    """
    common = schema_common(schema)
    template = _rsp.load_prompt(prompt_path or DEFAULT_PROMPT_PATHS[schema])
    with open(table_md_path, encoding="utf-8") as fh:
        table_text = fh.read()

    result = _rsp.curate_table(
        template, pmid, table_text, common,
        base_url=_rsp.resolve_base_url(base_url), model=model, temperature=temperature,
        num_ctx=num_ctx, timeout=timeout, call_fn=call_fn,
    )

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "arm_a.csv")
    common.write_csv(csv_path, result["rows"])
    with open(os.path.join(out_dir, "arm_a.raw.txt"), "w", encoding="utf-8") as fh:
        fh.write(result["raw"])
    if result.get("thinking"):  # the model's reasoning, for diagnosing empty or odd replies
        with open(os.path.join(out_dir, "arm_a.thinking.txt"), "w", encoding="utf-8") as fh:
            fh.write(result["thinking"])

    return {
        "csv_path": csv_path,
        "rows": len(result["rows"]),
        "dropped_rows": len(result["dropped"]),
        "seconds": result["elapsed_seconds"],
        "llm_calls": 2 if result["retried_no_think"] else 1,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }


_WORKFLOWS: dict[str, object] = {}


def get_expensive_workflow(schema: str, llm=None, llm2=None):
    """The built multi-agent workflow for `schema`, cached per process (building
    compiles the LangGraph; the LLM handles come from PIPELINE_LLM/AGENT_LLM
    the same way PKSummaryTask/PKIndividualTask wire their tools)."""
    wf = _WORKFLOWS.get(schema)
    if wf is None:
        if schema == "pk_summary":
            from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow
            wf = PKSumWorkflow(llm=llm or get_pipeline_llm())
        elif schema == "pk_individual":
            from extractor.agents.pk_individual.pk_ind_workflow import PKIndWorkflow
            wf = PKIndWorkflow(llm=llm or get_pipeline_llm(), llm2=llm2 or get_agent_llm())
        elif schema in _FULL_TEXT_WORKFLOWS or schema == "pe_study_outcome":
            module_name, class_name = (_FULL_TEXT_WORKFLOWS.get(schema)
                                       or ("extractor.agents.pe_study_outcome_ver2.pe_study_out_workflow",
                                           "PEStudyOutWorkflow"))
            cls = getattr(importlib.import_module(module_name), class_name)
            wf = cls(llm=llm or get_pipeline_llm())
        else:
            raise ValueError(f"no multi-agent workflow wired for schema {schema!r}")
        wf.build()
        _WORKFLOWS[schema] = wf
    return wf


# The workflow class each FullTextCurationTool / population tool instantiates
# (extractor/agents_manager/*_task.py). pe_study_outcome uses the ver2 workflow,
# the one PEStudyOutcomeCurationTool runs.
_FULL_TEXT_WORKFLOWS = {
    "pk_drug_summary": ("extractor.agents.pk_drug_summary.pk_drug_sum_workflow", "PKDrugSumWorkflow"),
    "pk_drug_individual": ("extractor.agents.pk_drug_individual.pk_drug_ind_workflow", "PKDrugIndWorkflow"),
    "pk_specimen_summary": ("extractor.agents.pk_specimen_summary.pk_spec_sum_workflow", "PKSpecSumWorkflow"),
    "pk_specimen_individual": ("extractor.agents.pk_specimen_individual.pk_spec_ind_workflow",
                               "PKSpecIndWorkflow"),
    "pk_population_summary": ("extractor.agents.pk_population_summary.pk_popu_sum_workflow",
                              "PKPopuSumWorkflow"),
    "pk_population_individual": ("extractor.agents.pk_population_individual.pk_popu_ind_workflow",
                                 "PKPopuIndWorkflow"),
    "pe_study_info": ("extractor.agents.pe_study_info.pe_study_info_workflow", "PEStudyInfoWorkflow"),
}


def verify_correct_table(pmid, schema, df, source_md, *, title=None, abstract=None,
                         full_text=None, llm=None, step=None, step_callback=None) -> dict:
    """The production per-table verify/correct loop on one curated table - the
    "multiagent" half of Arm B. Runs `PKPEPerTableVerifyCorrectStep._run_one_table`
    (the node PKPEManager's VerifyScopeEnum.PerTable uses) with the same state
    PKPEAgentToolTask._run_workflow seeds, scoped to this table only; the
    verification and correction steps and their prompts are the unmodified
    production ones. `llm` defaults to `get_agent_llm()` like the task does.

    Returns {"df": corrected DataFrame (workflow column names, as the loop saw
    them), "final_answer": FinalAnswerEnum name, "token_usage"}.
    """
    from TabFuncFlow.utils.table_utils import markdown_to_dataframe
    from extractor.agents.pk_pe_agents.pk_pe_agents_types import PaperTypeEnum
    from extractor.agents.pk_pe_agents.pk_pe_per_table_verify_step import PKPEPerTableVerifyCorrectStep

    step = step or PKPEPerTableVerifyCorrectStep(llm=llm or get_agent_llm(), pmid=pmid,
                                                 domain=_VERIFY_DOMAINS[schema])
    state = {
        "pmid": pmid,
        "paper_type": PaperTypeEnum.PK,
        "paper_title": title or "",
        "paper_abstract": abstract or "",
        "full_text": full_text,
        "step_output_callback": step_callback,
        "step_count": 0,
    }
    corrected_md, answer, token_usage = step._run_one_table(
        state, dataframe_to_markdown(df), source_md, table_label="Arm B verify/correct"
    )
    corrected = markdown_to_dataframe(corrected_md) if corrected_md else pd.DataFrame()
    return {"df": corrected, "final_answer": getattr(answer, "name", str(answer)),
            "token_usage": token_usage}


def run_expensive_arm_table(pmid, schema, table, out_dir, *, title=None, abstract=None,
                            full_text=None, mode=DEFAULT_ARM_B_MODE, workflow=None, llm=None,
                            llm2=None, step_callback=None, verify_correct_fn=None) -> dict:
    """Arm B on one unit (see module docstring for the two modes). `table` is a
    pmid_db table dict {"caption", "footnote", "table": DataFrame} for a table
    unit, or {"text": str, "table": None} for a full-text unit (the
    orchestrator builds both - see curate_orchestrator.build_units). Writes
    `<out_dir>/arm_b.csv` on the schema's canonical columns; in "multiagent"
    mode also `<out_dir>/arm_b.execution.csv`, the table before the loop.

    A table the workflow cannot curate (the tools' own "No valid entries
    found" case, or any other failure) yields an empty table and an "error"
    entry - one bad table must not abort the paper, same as the tools. The
    loop is skipped on an empty execution result (nothing to verify, exactly
    as PKPEPerTableVerifyCorrectStep skips tables that produced no data).

    Returns {"csv_path", "rows", "seconds", "error", "mode", "final_answer",
    "token_usage"}; the last two are None in "pipeline" mode.
    """
    if mode not in ARM_B_MODES:
        raise ValueError(f"unknown Arm B mode {mode!r}; expected one of {ARM_B_MODES}")
    workflow = workflow or get_expensive_workflow(schema, llm=llm, llm2=llm2)
    verify_correct_fn = verify_correct_fn or verify_correct_table
    is_table_unit = table.get("table") is not None
    caption = "\n".join([table.get("caption") or "", table.get("footnote") or ""])
    md_table = dataframe_to_markdown(table["table"]) if is_table_unit else None
    source_text = unit_source_text(table)  # what the verify/correct loop checks against

    if WORKFLOW_INPUT.get(schema, "md_table") == "md_table":
        if not is_table_unit:
            raise ValueError(f"schema {schema!r} curates tables; got a text unit")
        kwargs = dict(md_table=md_table, caption_and_footnote=caption, title=title,
                      step_callback=step_callback)
        if schema == "pk_individual":
            kwargs["full_text"] = full_text
        run = lambda: workflow.go_md_table(**kwargs)  # noqa: E731
    else:
        run = lambda: workflow.go_full_text(title=title or "", full_text=source_text,  # noqa: E731
                                            step_callback=step_callback)

    t0 = time.time()
    df, error, final_answer, token_usage = None, None, None, None
    try:
        df = run()
    except Exception as exc:  # noqa: BLE001 - mirror the tools: log, empty table, keep going
        logger.exception("Arm B failed on PMID %s, table %r", pmid, (table.get("caption") or "")[:80])
        error = str(exc)

    os.makedirs(out_dir, exist_ok=True)
    if mode == "multiagent" and df is not None and not df.empty:
        df.to_csv(os.path.join(out_dir, "arm_b.execution.csv"), index=False)
        try:
            vc = verify_correct_fn(pmid, schema, df, source_text, title=title, abstract=abstract,
                                   full_text=full_text, llm=llm2, step_callback=step_callback)
            df, final_answer, token_usage = vc["df"], vc["final_answer"], vc["token_usage"]
        except Exception as exc:  # noqa: BLE001 - keep the execution result, same as a failed correction
            logger.exception("Arm B verify/correct loop failed on PMID %s, table %r", pmid,
                             (table.get("caption") or "")[:80])
            error = str(exc)
            final_answer = "VerificationError"
    seconds = time.time() - t0

    csv_path = os.path.join(out_dir, "arm_b.csv")
    if df is not None and not df.empty:
        df = drop_placeholder_rows(normalize_to_schema(df, schema))
    if df is None or df.empty:
        write_empty_csv(csv_path, schema)
        rows = 0
    else:
        df.to_csv(csv_path, index=False)
        rows = len(df)

    return {"csv_path": csv_path, "rows": rows, "seconds": seconds, "error": error,
            "mode": mode, "final_answer": final_answer, "token_usage": token_usage}
