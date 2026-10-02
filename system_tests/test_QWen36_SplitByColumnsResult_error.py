"""Reproduce / diagnose: qwen3.6 returns a bare JSON list for SplitByColumnsResult.

Symptom (pipeline mode, PMID 10971311 Table 1, qwen3.6:35b-t0 via Ollama):

    OutputParserException: Failed to parse SplitByColumnsResult from completion
    [["('Unnamed: 0_level_0', 'Volunteer')", ...], [...]]
    Input should be a valid dictionary or instance of SplitByColumnsResult

i.e. the model emits a top-level JSON *array* where the schema requires the
*object* {"sub_tables_columns": [[...], ...]}. gpt-4o never does this.

Already ruled out (see the debugging notes in the conversation history):
  * presence_penalty (base qwen3.6 ships 1.5). Rebuilt qwen3.6:35b-t0 with
    presence_penalty=0, confirmed effective in Ollama's request merge, and the
    identical failure still happened.

Hypotheses this file lets you separate (run it in an interactive GPU session):

  H1. Ollama is NOT enforcing the `format=<json schema>` grammar for this model.
      Evidence so far: the header-categorisation call, which *succeeded*, still
      came back wrapped in a ```json fence - impossible under real grammar
      enforcement - and the failing call started with '[' - impossible under an
      object schema. => test_format_schema_is_enforced_minimal
  H2. The task prompt contradicts the schema. SPLIT_BY_COLUMNS_PROMPT tells the
      model "Return the results as a list of lists ... [[...],[...]]" (a bare-list
      example) while the schema wants an object. With no grammar to force the
      wrapper, the model follows the concrete example.
      => test_prompt_variants_matrix (legacy bare_list variants: the prompt is now fixed, so these rows should be the only ones that fail)
  H3. Something about the request shape: streaming (ChatOllama always streams),
      `think` handling, or the pipeline sending a *system-only* message list
      (no user turn). => test_prompt_variants_matrix (stream / think / user-turn)

Tests:
  test_classify_output_helper                     offline sanity of the classifier
  test_prompt_matches_object_schema               offline regression guard for H2 (fixed)
  test_agent_fix_parser_wraps_bare_list           offline check of the bare-list fallback
  test_format_schema_is_enforced_minimal          HARD assert; fails => H1
  test_split_by_columns_step_pipeline_path        HARD assert; the real repro
  test_prompt_variants_matrix                     diagnostic table, no conformance assert

How to run (interactive desktop / GPU node):

    # 1. serve the model with the *old* Ollama build (ollama.sif v0.33 has an
    #    unrelated CUDA crash bug with qwen3.6 - see TODO/complete-benchmark-results.md)
    apptainer exec --nv \\
        --env OLLAMA_MODELS=/users/PCON0100/feng1426/ollama/models \\
        --env OLLAMA_HOST=127.0.0.1:11434 --env OLLAMA_CONTEXT_LENGTH=65536 \\
        /users/PCON0100/feng1426/ollama/ollama-old.sif ollama serve &

    # 2. run
    module load miniconda3/24.1.2-py310 && conda activate tabular-data
    cd /users/PCON0100/feng1426/projects/github/tabular-data
    python -m pytest system_tests/test_QWen36_SplitByColumnsResult_error.py -v

Environment variables (all optional):
    OLLAMA_BASE_URL          default http://127.0.0.1:11434
    QWEN_TEST_MODEL          default qwen3.6:35b-t0   (model under test)
    QWEN_TEST_CONTROL_MODEL  default qwen3.8:27b-t0   (comparison; skipped if absent)
    QWEN_TEST_REPEATS        default 2                (calls per matrix config)
    QWEN_TEST_OUT            if set, matrix results / prompts are dumped there as JSON/txt

Everything that needs a server is skipped cleanly when none is reachable, so this
file is safe to leave in system_tests/.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import pytest
import requests
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from extractor.agents.agent_factory import (
    MAX_PIPELINE_AGENT_CONTENT_NUM,
    MAX_PIPELINE_AGENT_PREDICT_NUM,
)
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.common_agent.common_agent_ollama import (
    IMPORTANT_INSTRUCTIONS,
    CommonAgentOllama,
)
from extractor.agents.pk_individual.pk_ind_split_by_col_agent import (
    SplitByColumnsResult,
    get_split_by_columns_prompt,
)
from extractor.agents.pk_individual.pk_ind_split_by_col_step import SplitByColumnsStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState
from extractor.llm_utils import get_format_instructions
from extractor.prompts_utils import generate_previous_errors_prompt
from extractor.utils import escape_braces_for_format
from system_tests.conftest_data_10971311 import data_md_table_aligned_10971311_table_0

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("QWEN_TEST_MODEL", "qwen3.6:35b-t0")
CONTROL_MODEL = os.environ.get("QWEN_TEST_CONTROL_MODEL", "qwen3.8:27b-t0")
REPEATS = int(os.environ.get("QWEN_TEST_REPEATS", "2"))
OUT_DIR = os.environ.get("QWEN_TEST_OUT")

# Same context / generation limits the pipeline's tool LLM uses
# (agent_factory.get_pipeline_llm -> get_gpt_qwen36_35b).
NUM_CTX = MAX_PIPELINE_AGENT_CONTENT_NUM
NUM_PREDICT = MAX_PIPELINE_AGENT_PREDICT_NUM

# --------------------------------------------------------------------------- #
# the exact failing input: PMID 10971311, Table 1 (aligned table + col mapping,
# both taken from the failing pipeline_qwen36 log)
# --------------------------------------------------------------------------- #
MD_TABLE_ALIGNED = data_md_table_aligned_10971311_table_0.strip()
COL_MAPPING = {
    "('Unnamed: 0_level_0', 'Volunteer')": "Patient ID",
    "('Unnamed: 1_level_0', 'tmax (h)')": "Parameter value",
    "('Citalopram', 'Maximum1 milk concentration (µg l−1)')": "Parameter value",
    "('Citalopram', 'Average2 milk concentration (µg l−1)')": "Parameter value",
    "('Unnamed: 4_level_0', 'M/PAUC')": "Parameter value",
    "('Unnamed: 5_level_0', 'tmax (h)')": "Parameter value",
    "('Demethylcitalopram', 'Maximum milk concentration (µg l−1)')": "Parameter value",
    "('Demethylcitalopram', 'Average milk concentration (µg l−1)')": "Parameter value",
    "('Unnamed: 8_level_0', 'M/PAUC')": "Parameter value",
}

# The legacy bare-list instruction that SPLIT_BY_COLUMNS_PROMPT used to contain
# (H2; now fixed to the object form below). The "bare_list" variants re-inject it
# so the matrix can still show the difference.
BARE_LIST_BLOCK = (
    "Return the results as a list of lists, where each inner list represents a "
    "sub-table with its included columns.\n"
    '[["ColumnA", "ColumnB", "ColumnC", "ColumnG"], '
    '["ColumnA", "ColumnD", "ColumnE", "ColumnF", "ColumnG"]]'
)
OBJECT_BLOCK = (
    'Return the results as a JSON object with a single key "sub_tables_columns" '
    "whose value is a list of lists, where each inner list represents a sub-table "
    "with its included columns.\n"
    '{"sub_tables_columns": [["ColumnA", "ColumnB", "ColumnC", "ColumnG"], '
    '["ColumnA", "ColumnD", "ColumnE", "ColumnF", "ColumnG"]]}'
)


# --------------------------------------------------------------------------- #
# prompt construction - mirrors CommonAgentOllama._invoke_agent line for line
# --------------------------------------------------------------------------- #
def build_task_prompt(variant: str = "pipeline") -> str:
    """The step-level prompt (SplitByColumnsStep.get_system_prompt)."""
    task = get_split_by_columns_prompt(MD_TABLE_ALIGNED, COL_MAPPING)
    task += generate_previous_errors_prompt("N/A")
    if variant == "bare_list":
        assert OBJECT_BLOCK in task, (
            "SPLIT_BY_COLUMNS_PROMPT changed - update OBJECT_BLOCK in this test, "
            "otherwise the 'bare_list' variant silently equals the baseline"
        )
        task = task.replace(OBJECT_BLOCK, BARE_LIST_BLOCK)
    elif variant != "pipeline":
        raise ValueError(variant)
    return task


def build_system_prompt(variant: str = "pipeline") -> str:
    """The final system message text the Ollama agent actually sends."""
    system_prompt = escape_braces_for_format(build_task_prompt(variant))
    format_instructions = get_format_instructions(SplitByColumnsResult)
    format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    system_prompt = system_prompt + "\n\n" + format_instructions
    system_prompt = system_prompt + "\n\n" + IMPORTANT_INSTRUCTIONS
    system_prompt = system_prompt + "\n\n/no_think"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    return prompt.format_messages(input=INSTRUCTION_PROMPT)[0].content


# --------------------------------------------------------------------------- #
# output classification
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def classify_output(content: Optional[str]) -> dict[str, Any]:
    """Describe a raw model reply the way the pipeline will experience it.

    fenced     - reply starts with a ``` fence (impossible under real grammar enforcement)
    first_char - first non-space char of the raw reply ('{' expected under an object schema)
    json_type  - top-level JSON type after removing fences: dict / list / invalid-json / empty
    valid      - does the pipeline's own parse chain accept it as a SplitByColumnsResult?
    """
    if content is None or not content.strip():
        return {"fenced": False, "first_char": "", "json_type": "empty", "valid": False,
                "error": "empty reply"}
    fenced = content.lstrip().startswith("```")
    first_char = content.lstrip()[:1]
    try:
        json_type = type(json.loads(_strip_fences(content))).__name__
    except Exception:
        json_type = "invalid-json"
    # Exactly what runnable_agent does: handle_qwen_thinking, then the parser.
    cleaned = CommonAgentOllama.handle_qwen_thinking(content)
    try:
        PydanticOutputParser(pydantic_object=SplitByColumnsResult).parse(cleaned)
        valid, error = True, ""
    except Exception as exc:  # noqa: BLE001 - we want the message
        valid, error = False, str(exc).splitlines()[0][:160]
    return {"fenced": fenced, "first_char": first_char, "json_type": json_type,
            "valid": valid, "error": error}


# --------------------------------------------------------------------------- #
# raw Ollama access (bypasses LangChain so request shape is fully explicit)
# --------------------------------------------------------------------------- #
def _list_models() -> Optional[set[str]]:
    try:
        r = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:  # noqa: BLE001
        return None


def chat(
    model: str,
    messages: list[dict],
    fmt: Any = None,
    think: Optional[bool] = False,
    stream: bool = True,
) -> dict[str, Any]:
    """POST /api/chat shaped like ChatOllama._chat_params (options, stream default True)."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        },
        "keep_alive": "10m",
    }
    if fmt is not None:
        payload["format"] = fmt
    if think is not None:
        payload["think"] = think

    t0 = time.time()
    content, thinking, last = "", "", {}
    if stream:
        with requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=900, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message") or {}
                content += msg.get("content") or ""
                thinking += msg.get("thinking") or ""
                last = chunk
    else:
        r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=900)
        r.raise_for_status()
        last = r.json()
        msg = last.get("message") or {}
        content = msg.get("content") or ""
        thinking = msg.get("thinking") or ""
    return {
        "content": content,
        "thinking": thinking,
        "done_reason": last.get("done_reason"),
        "eval_count": last.get("eval_count"),
        "prompt_eval_count": last.get("prompt_eval_count"),
        "elapsed": round(time.time() - t0, 1),
    }


def _maybe_dump(name: str, payload: Any) -> None:
    if not OUT_DIR:
        return
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ollama_models() -> set[str]:
    models = _list_models()
    if models is None:
        pytest.skip(
            f"No Ollama server reachable at {BASE_URL}. Start one (see the module "
            "docstring) or set OLLAMA_BASE_URL."
        )
    return models


def _resolve_model(kind: str, models: set[str]) -> str:
    model = MODEL if kind == "qwen36" else CONTROL_MODEL
    if model not in models:
        pytest.skip(f"{kind} model {model!r} not served at {BASE_URL}; available: {sorted(models)}")
    return model


def _make_llm(model: str) -> ChatOllama:
    """Same construction as agent_factory.get_gpt_qwen36_35b(max_content_num, max_predict_num)."""
    return ChatOllama(
        base_url=BASE_URL,
        model=model,
        reasoning=False,
        streaming=True,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        timeout=900,
    )


def _step_callback(
    step_name=None, step_description=None, step_output=None,
    step_reasoning_process=None, token_usage=None,
):
    if step_name:
        logger.info("== %s ==", step_name)
    if step_output:
        logger.info("%s", step_output)


# --------------------------------------------------------------------------- #
# offline tests (no server needed)
# --------------------------------------------------------------------------- #
def test_classify_output_helper():
    good = '{"sub_tables_columns": [["a", "b"], ["a", "c"]]}'
    fenced_good = "```json\n" + good + "\n```"
    bare_list = '[["a", "b"], ["a", "c"]]'

    r = classify_output(good)
    assert (r["fenced"], r["first_char"], r["json_type"], r["valid"]) == (False, "{", "dict", True)

    r = classify_output(fenced_good)
    assert r["fenced"] is True and r["json_type"] == "dict" and r["valid"] is True

    r = classify_output(bare_list)
    assert (r["fenced"], r["first_char"], r["json_type"], r["valid"]) == (False, "[", "list", False)
    assert "valid dictionary" in r["error"] or "Failed to parse" in r["error"]

    assert classify_output("")["json_type"] == "empty"
    assert classify_output("not json at all")["json_type"] == "invalid-json"


def test_prompt_matches_object_schema():
    """Regression guard for H2: the prompt example must be the object the schema wants."""
    schema = SplitByColumnsResult.model_json_schema()
    assert schema["type"] == "object"
    assert "sub_tables_columns" in schema["properties"]

    prompt = build_system_prompt("pipeline")
    _maybe_dump("prompt_pipeline.txt", prompt)
    assert BARE_LIST_BLOCK not in prompt, "the prompt again shows a bare-list example"
    assert '{"sub_tables_columns"' in prompt, "prompt lacks the object example"

    legacy = build_system_prompt("bare_list")
    _maybe_dump("prompt_bare_list.txt", legacy)
    assert BARE_LIST_BLOCK in legacy and OBJECT_BLOCK not in legacy


def test_agent_fix_parser_wraps_bare_list():
    """The fallback recovers a bare list of lists, and only that."""
    from extractor.agents.pk_individual.pk_ind_split_by_col_agent import (
        agent_fix_parser_split_by_columns as fix,
    )

    res = fix('[["a", "b"], ["a", "c"]]')
    assert res is not None and res.sub_tables_columns == [["a", "b"], ["a", "c"]]
    res = fix('```json\n[["a", "b"]]\n```')
    assert res is not None and res.sub_tables_columns == [["a", "b"]]
    assert fix('{"other": 1}') is None
    assert fix('[1, 2]') is None
    assert fix("not json") is None


# --------------------------------------------------------------------------- #
# H1: is the `format` json-schema grammar enforced at all?  (HARD assertion)
# --------------------------------------------------------------------------- #
PROBE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}
# Deliberately asks for something the schema forbids (a bare array). If Ollama
# enforces the grammar, the reply MUST still be {"answer": ...}.
PROBE_MESSAGES = [
    {
        "role": "user",
        "content": 'List three colours. Reply with a JSON array like ["red", "green", "blue"] '
        "and nothing else.",
    }
]


@pytest.mark.parametrize("think", [False, None], ids=["think_false", "think_unset"])
@pytest.mark.parametrize("model_key", ["qwen36", "control"])
def test_format_schema_is_enforced_minimal(model_key, think, ollama_models):
    """Fails => Ollama returned schema-violating output despite `format=<schema>` (H1).

    If qwen3.6 fails and the control passes: model/architecture-specific.
    If both fail: this Ollama build isn't enforcing `format` schemas (or `think`
    interacts with it) - not a qwen3.6 problem per se.
    """
    model = _resolve_model(model_key, ollama_models)
    res = chat(model, PROBE_MESSAGES, fmt=PROBE_SCHEMA, think=think, stream=True)
    content = res["content"]
    print(f"\n[{model} think={think}] reply: {content[:300]!r}")

    try:
        parsed = json.loads(content)  # no fence stripping: enforced output is bare JSON
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"format schema NOT enforced for {model} (think={think}): reply is not raw JSON "
            f"({exc}). reply={content[:300]!r}"
        )
    assert isinstance(parsed, dict) and "answer" in parsed, (
        f"format schema NOT enforced for {model} (think={think}): got a "
        f"{type(parsed).__name__} instead of {{'answer': ...}}. reply={content[:300]!r}"
    )


# --------------------------------------------------------------------------- #
# the real repro through the actual step / agent code (HARD assertion)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_key", ["qwen36", "control"])
def test_split_by_columns_step_pipeline_path(model_key, ollama_models, caplog):
    """Runs SplitByColumnsStep exactly as PKIndividualTablesCurationTool does.

    Expected today: qwen3.6 fails (OutputParserException -> tenacity RetryError after
    5 attempts); a control model that follows the schema passes.
    """
    model = _resolve_model(model_key, ollama_models)
    caplog.set_level(logging.INFO, logger="extractor.agents.common_agent.common_agent_ollama")

    state = PKIndWorkflowState()
    state["llm"] = _make_llm(model)
    state["col_mapping"] = dict(COL_MAPPING)  # leave_step mutates it
    state["md_table_aligned"] = MD_TABLE_ALIGNED
    state["step_callback"] = _step_callback

    try:
        SplitByColumnsStep().execute(state)
    except Exception as exc:  # noqa: BLE001
        raws = [
            r.getMessage() for r in caplog.records
            if "raw.content preview" in r.getMessage()
        ]
        shown = "\n".join(f"  attempt {i + 1}: {m[:420]}" for i, m in enumerate(raws[:3]))
        pytest.fail(
            f"SplitByColumnsStep failed for {model}: {type(exc).__name__}: {str(exc)[:200]}\n"
            f"raw model replies seen by the parser ({len(raws)} attempts):\n{shown}"
        )

    md_table_list = state.get("md_table_list")
    assert isinstance(md_table_list, list) and md_table_list, "no sub-tables produced"


# --------------------------------------------------------------------------- #
# H1/H2/H3 side by side - diagnostic, prints a table (no conformance assertion)
# --------------------------------------------------------------------------- #
# (name, prompt variant, format, think, add user turn, stream)
CONFIGS = [
    ("baseline (as pipeline)",         "pipeline",       "schema", False, False, True),
    ("non-streaming",                  "pipeline",       "schema", False, False, False),
    ("no format",                      "pipeline",       None,     False, False, True),
    ('format="json"',                  "pipeline",       "json",   False, False, True),
    ("think unset",                    "pipeline",       "schema", None,  False, True),
    ("+ user turn",                    "pipeline",       "schema", False, True,  True),
    ("legacy bare-list prompt",        "bare_list",      "schema", False, False, True),
    ("legacy bare-list, no format",    "bare_list",      None,     False, False, True),
    ("legacy bare-list + user turn",   "bare_list",      "schema", False, True,  True),
]


def _messages(system_prompt: str, add_user_turn: bool) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    if add_user_turn:
        msgs.append({"role": "user", "content": INSTRUCTION_PROMPT})
    return msgs


def _print_table(model: str, rows: list[dict]) -> None:
    cols = ["config", "rep", "fenced", "1st", "json_type", "valid", "secs", "eval_tok"]
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(f"\n=== {model} - SplitByColumnsResult, PMID 10971311 Table 1 ===")
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    print(
        "\nHow to read it:\n"
        "  valid=False + json_type=list        -> the bare-list failure being debugged\n"
        "  fenced=True on 'schema' rows        -> `format` grammar not enforced (H1)\n"
        "  'legacy bare-list' rows invalid, baseline valid -> the prompt contradiction was the cause (H2)\n"
        "  '+ user turn' / non-streaming differ -> request shape matters (H3)\n"
    )


@pytest.mark.parametrize("model_key", ["qwen36", "control"])
def test_prompt_variants_matrix(model_key, ollama_models, capsys):
    model = _resolve_model(model_key, ollama_models)
    prompts = {v: build_system_prompt(v) for v in ("pipeline", "bare_list")}

    rows: list[dict] = []
    raw_records: list[dict] = []
    for name, variant, fmt_kind, think, add_user, stream in CONFIGS:
        fmt = {"schema": SplitByColumnsResult.model_json_schema(), "json": "json", None: None}[fmt_kind]
        for rep in range(1, REPEATS + 1):
            res = chat(model, _messages(prompts[variant], add_user), fmt=fmt, think=think, stream=stream)
            info = classify_output(res["content"])
            rows.append({
                "config": name, "rep": rep,
                "fenced": info["fenced"], "1st": info["first_char"] or "-",
                "json_type": info["json_type"], "valid": info["valid"],
                "secs": res["elapsed"], "eval_tok": res["eval_count"],
            })
            raw_records.append({"config": name, "rep": rep, **info, **res})

    with capsys.disabled():
        _print_table(model, rows)
    _maybe_dump(f"matrix_{model_key}.json", {"model": model, "records": raw_records})

    # Conformance is deliberately NOT asserted here: this test exists to print the table.
    assert rows, "no matrix rows collected"
