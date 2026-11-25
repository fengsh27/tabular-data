"""Tests for scripts/simple_prompt/run_simple_prompt.py's per-table unit,
`curate_table`, which the CurateAgent cascade's Arm A runs in-process and the
script's own main() loops over."""

import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "simple_prompt"))

import run_simple_prompt as rsp  # noqa: E402

common = importlib.import_module("_common_summary")
HEADER = ",".join(common.COLS)
ROW = ",".join(["Ketamine", "Ketamine", "Plasma", "children", "", "19",
                "AUC", "91", "Mg/min/L", "mean", "SD", "30", "", "", "", ""])
TEMPLATE = "PMID {PMID}\n\n{TABLE}\n\nReturn CSV."


def _recorder(responses):
    calls = []

    def call_fn(prompt, **kw):
        calls.append({"prompt": prompt, **kw})
        body = responses[min(len(calls) - 1, len(responses) - 1)]
        return {"prompt_eval_count": 11, "eval_count": 4, "thinking": "", **body}
    call_fn.calls = calls
    return call_fn


def test_render_prompt_fills_both_placeholders():
    out = rsp.render_prompt(TEMPLATE, "123", "| a |")
    assert "PMID 123" in out and "| a |" in out
    assert "{TABLE}" not in out and "{PMID}" not in out


def test_curate_table_parses_rows_and_reports_tokens():
    call_fn = _recorder([{"response": f"{HEADER}\n{ROW}\n"}])

    res = rsp.curate_table(TEMPLATE, "123", "| a |", common, call_fn=call_fn)

    assert len(res["rows"]) == 1 and res["rows"][0]["Value"] == "91"
    assert res["dropped"] == []
    assert res["retried_no_think"] is False
    assert (res["input_tokens"], res["output_tokens"]) == (11, 4)
    assert res["elapsed_seconds"] >= 0
    assert len(call_fn.calls) == 1 and "think" not in call_fn.calls[0]


def test_empty_response_is_retried_once_without_thinking():
    call_fn = _recorder([{"response": "", "thinking": "..."}, {"response": f"{HEADER}\n{ROW}\n"}])

    res = rsp.curate_table(TEMPLATE, "123", "| a |", common, call_fn=call_fn)

    assert res["retried_no_think"] is True
    assert len(res["rows"]) == 1
    assert [c.get("think") for c in call_fn.calls] == [None, False]


def test_retry_can_be_disabled_for_non_thinking_backends():
    call_fn = _recorder([{"response": ""}])

    res = rsp.curate_table(TEMPLATE, "123", "| a |", common, call_fn=call_fn,
                           retry_without_think=False)

    assert res["rows"] == [] and res["retried_no_think"] is False
    assert len(call_fn.calls) == 1


def test_unparsable_rows_are_counted_as_dropped():
    call_fn = _recorder([{"response": f"{HEADER}\n{ROW}\nonly,three,fields\n"}])

    res = rsp.curate_table(TEMPLATE, "123", "| a |", common, call_fn=call_fn)

    assert len(res["rows"]) == 1
    assert len(res["dropped"]) == 1


def test_header_only_reply_is_retried_without_thinking_and_thinking_is_kept(tmp_path):
    """qwen3.8 on pk_summary (2026-10-31): thousands of thinking tokens, then a header
    line and nothing else. That is retried with think=False like an empty reply."""
    import importlib, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "simple_prompt"))
    rsp = importlib.import_module("run_simple_prompt")
    summary = importlib.import_module("_common_summary")
    header = ",".join(summary.COLS)
    good = ",".join(["Ketamine", "Ketamine", "", "children", "", "", "Cmax", "1.6", "mg/L", "mean", "SD", "0.68", "", "", "", ""])
    calls = []

    def call_fn(prompt, **kw):
        calls.append(kw)
        if kw.get("think") is False:
            return {"response": f"{header}\n{good}\n", "prompt_eval_count": 5, "eval_count": 9}
        return {"response": header, "thinking": "let me draft the rows ...", "prompt_eval_count": 5, "eval_count": 3506}

    res = rsp.curate_table("T", "1", "| a |", summary, call_fn=call_fn)

    assert calls == [{}, {"think": False}]
    assert res["retried_no_think"] is True and len(res["rows"]) == 1 and res["rows"][0]["Value"] == "1.6"
    assert res["thinking"] == "let me draft the rows ..."


def test_a_reply_with_rows_is_not_retried():
    import importlib, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "simple_prompt"))
    rsp = importlib.import_module("run_simple_prompt")
    summary = importlib.import_module("_common_summary")
    good = ",".join(["Ketamine", "Ketamine", "", "children", "", "", "Cmax", "1.6", "mg/L", "mean", "SD", "0.68", "", "", "", ""])
    calls = []

    def call_fn(prompt, **kw):
        calls.append(kw)
        return {"response": f"{','.join(summary.COLS)}\n{good}\n", "prompt_eval_count": 5, "eval_count": 9}

    res = rsp.curate_table("T", "1", "| a |", summary, call_fn=call_fn)
    assert calls == [{}] and res["retried_no_think"] is False and len(res["rows"]) == 1
