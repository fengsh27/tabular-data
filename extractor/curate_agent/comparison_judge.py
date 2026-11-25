"""Tier 4 - LLM comparison of two candidate curated tables for one source table.

Per docs/VERIFICATION_AGENT_DESIGN.md's ComparisonJudge. In the v2 per-table
cascade (`curate_orchestrator._curate_table`) this runs whenever Arm B ran,
i.e. whenever Arm A's table failed the Tier 1-3 gate - not only on a tie.
The prompts therefore no longer claim the error rates are tied, do not show
the composite error_rate at all (see `_format_report`), and state that an
empty candidate is correct when the source table has no curatable data
(`TIER4_JUDGE_RULES`) - the three defects the first live round's Tier 4 audit
found (docs/cascade_mode_results.md).

Reuses the `PKPECommonStep`/`get_common_agent()` pattern and a prompt
structurally similar to `PKPECuratedTablesVerificationStep`
(`extractor/agents/pk_pe_agents/pk_pe_verification_step.py`), but judging
**two** candidate tables against the source rather than one table in
isolation.

Two LLM calls, not one, to honor the design's "not the raw tables first -
only pull full tables into the prompt if the reports alone can't
disambiguate": the first pass sees only the two ValidityReports and paper
context (cheap); a second pass, with the actual curated tables attached,
only runs if the first says it isn't confident.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from extractor.agents.agent_factory import get_agent_llm
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE
from extractor.agents.pk_pe_agents.pk_pe_common_step import PKPECommonStep
from extractor.constants import COT_USER_INSTRUCTION

logger = logging.getLogger(__name__)

TIER4_JUDGE_RULES = """\
* Candidate A is a single-pass curation (one model call on the source - a table, or the paper's full text for the full-text table types); Candidate B is a multi-agent workflow's curation of the same source. Neither is right by default.
* **A candidate with zero rows is CORRECT when the source contains no data of the target table type** (e.g. only group means for an individual-level table, cases from other cited studies, or a paper with no dosing for a dosing table). Zero rows is NOT a failure by itself - use the source assessment given in that candidate's report, and prefer the empty candidate over one that filled a non-curatable source with rows.
* The two reports were produced by the same checks but are not on one scale (Candidate A's structural check does not apply to Candidate B's output format), so compare the findings, not a single number."""

TIER4_REPORTS_ONLY_PROMPT = """
You are a biomedical data verification assistant with expertise in {domain}.
Two independent pipelines each curated the same source (a table of a paper, or
the paper's full text) into a data table (Candidate A and Candidate B). A
validity check ran on both and produced the two reports below. You are being asked to judge which candidate
is actually more accurate, using only the paper context and the two reports -
the actual curated tables are NOT included in this pass, to save cost.

### **Rules**

{rules}

---

### **Paper Title**

{paper_title}

### **Paper Abstract**

{paper_abstract}

### **Candidate A's validity report**

{report_a}

### **Candidate B's validity report**

{report_b}

---

### **Your Output**

Respond using the **exact json compact format** below:

```
{{
  "confident": <boolean, True only if the reports above are enough to judge which candidate is more accurate>,
  "winner": <"A" or "B" or null, your pick if confident, otherwise null>,
  "explanation": <string, max 150 words, your reasoning>
}}
```

If the two reports look substantively equivalent (e.g. both zero error, or
the same kind of finding on both sides) and you cannot honestly tell which
candidate is better from this alone, set `"confident": false` and leave
`"winner"` null - you will be shown the actual tables next.
"""

TIER4_FULL_PROMPT = """
You are a biomedical data verification assistant with expertise in {domain}.
Two independent pipelines each curated the same source (a table of a paper, or
the paper's full text) into a data table (Candidate A and Candidate B). A
validity check ran on both, and a first pass using only the two reports could
not decide between them. Judge which candidate is a more accurate, faithful
curation of the source below.

### **Rules**

{rules}

---

### **Paper Title**

{paper_title}

### **Paper Abstract**

{paper_abstract}

### **Source Table(s) or Text**

{source_tables}

### **Candidate A's validity report**

{report_a}

### **Candidate A's curated table**

{table_a}

### **Candidate B's validity report**

{report_b}

### **Candidate B's curated table**

{table_b}

---

### **Your Output**

Respond using the **exact json compact format** below:

```
{{
  "winner": <"A" or "B">,
  "explanation": <string, max 200 words, your reasoning - cite specific rows/values that decided it>
}}
```

You must pick one, even if the difference is small - say so in the
explanation rather than refusing to choose.
"""


class ComparisonReportsOnlyResult(BaseModel):
    confident: bool = Field(
        description="True only if the two validity reports and paper context "
                    "are enough to judge which candidate is more accurate."
    )
    winner: Optional[Literal["A", "B"]] = Field(
        default=None, description="Pick if confident, otherwise null."
    )
    explanation: str = Field(description="Reasoning, max 150 words.")


class ComparisonFullResult(BaseModel):
    winner: Literal["A", "B"] = Field(
        description="Which candidate table is more accurate and faithful to the source."
    )
    explanation: str = Field(description="Reasoning for the verdict, max 200 words.")


def _format_report(report: dict) -> str:
    """Human-readable summary of a ValidityReport, for the prompt.

    Per-tier findings only. The composite `error_rate` is deliberately NOT
    printed: Candidate A's includes Tier 1 and Candidate B's excludes it (see
    validity_checker.check_validity), and in the first live round the judge
    did nothing but compare the two numbers - 6 verdicts, all "lower
    error_rate wins", 3 of them wrong against gold.

    An empty table is described by its source-table assessment
    (`tier3["empty_table_justified"]`, from `llm_verification.assess_empty_table`)
    rather than by a meaningless fit score.
    """
    tier1 = report.get("tier1") or {}
    tier3 = report.get("tier3")
    total_rows = tier1.get("total_rows")
    lines = [f"- total_rows: {total_rows}"]
    if total_rows == 0:
        if tier3 and "empty_table_justified" in tier3:
            verdict = ("the source table has NO curatable data of this type - an empty table is the correct output"
                       if tier3["empty_table_justified"]
                       else "the source table DOES have curatable data - this empty table missed it")
            lines.append(f"- source-table assessment: {verdict}")
            lines.append(f"- assessment explanation: {tier3.get('explanation')}")
            for m in tier3.get("expected_rows") or []:
                lines.append(f"- row that should have been curated: {m}")
        else:
            lines.append("- source-table assessment: not available")
        return "\n".join(lines)

    lines.append(f"- dropped/implausible rows: {tier1.get('dropped_rows')}")
    lines.append(
        f"- avg per-row fit score: {tier1.get('avg_fit_score'):.2f}"
        if tier1.get("avg_fit_score") is not None else "- avg per-row fit score: N/A"
    )
    tier2 = report.get("tier2")
    if tier2:
        lines.append(
            f"- existence check: {tier2.get('unsupported')} unsupported "
            f"out of {tier2.get('checked_numbers')} numbers checked"
        )
    if tier3:
        lines.append(
            f"- LLM verification: correct={tier3.get('correct')}, "
            f"{tier3.get('error_row_count')} error row(s), "
            f"{tier3.get('missed_row_count')} missed row(s)"
        )
    return "\n".join(lines)


class ComparisonJudgeStep(PKPECommonStep):
    """Standalone Tier 4 step. Not part of PKPEManager's verify/correct loop -
    this runs only inside the CurateAgent cascade's tie-break case."""

    def __init__(self, llm: BaseChatOpenAI, domain: str = "pharmacokinetics"):
        super().__init__(llm)
        self.step_name = "Curate Agent Comparison Judge Step"
        self.domain = domain

    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        agent = self.get_agent(llm=self.llm)
        total_usage = {**DEFAULT_TOKEN_USAGE}

        if state.get("force_full_tables"):
            return self._full_pass(agent, state, total_usage)

        reports_prompt = TIER4_REPORTS_ONLY_PROMPT.format(
            domain=self.domain,
            rules=TIER4_JUDGE_RULES,
            paper_title=state["paper_title"],
            paper_abstract=state["paper_abstract"],
            report_a=_format_report(state["report_a"]),
            report_b=_format_report(state["report_b"]),
        )
        res, _, token_usage, reasoning = agent.go(
            system_prompt=reports_prompt,
            instruction_prompt=COT_USER_INSTRUCTION,
            schema=ComparisonReportsOnlyResult,
        )
        total_usage = {k: total_usage.get(k, 0) + token_usage.get(k, 0) for k in total_usage}
        self._print_step(state, step_output=reasoning or "N/A")

        if res.confident and res.winner is not None:
            state["winner"] = res.winner
            state["explanation"] = res.explanation
            state["used_full_tables"] = False
            return state, total_usage

        return self._full_pass(agent, state, total_usage)

    def _full_pass(self, agent, state, total_usage):
        full_prompt = TIER4_FULL_PROMPT.format(
            domain=self.domain,
            rules=TIER4_JUDGE_RULES,
            paper_title=state["paper_title"],
            paper_abstract=state["paper_abstract"],
            source_tables=state["source_tables"],
            report_a=_format_report(state["report_a"]),
            table_a=state["table_a"],
            report_b=_format_report(state["report_b"]),
            table_b=state["table_b"],
        )
        res, _, token_usage, reasoning = agent.go(
            system_prompt=full_prompt,
            instruction_prompt=COT_USER_INSTRUCTION,
            schema=ComparisonFullResult,
        )
        total_usage = {k: total_usage.get(k, 0) + token_usage.get(k, 0) for k in total_usage}
        self._print_step(state, step_output=reasoning or "N/A")

        state["winner"] = res.winner
        state["explanation"] = res.explanation
        state["used_full_tables"] = True
        return state, total_usage


def compare(paper_title, paper_abstract, source_tables,
            report_a, table_a, report_b, table_b,
            llm=None, domain="pharmacokinetics",
            step_output_callback=None, force_full_tables=False) -> dict:
    """Judge which of two candidates is more accurate.

    Returns {"winner": "A" | "B", "explanation": str, "used_full_tables": bool}.
    `force_full_tables=True` skips the reports-only pass and judges from the
    source table and both curated tables directly (one call instead of up to
    two). `llm` defaults to `get_agent_llm()` (the AGENT_LLM env var), same as
    the rest of the PKPE agent machinery.
    """
    step = ComparisonJudgeStep(llm=llm or get_agent_llm(), domain=domain)
    state = {
        "step_output_callback": step_output_callback,
        "paper_title": paper_title,
        "paper_abstract": paper_abstract,
        "source_tables": source_tables,
        "report_a": report_a,
        "table_a": table_a,
        "report_b": report_b,
        "table_b": table_b,
        "force_full_tables": force_full_tables,
    }
    state = step.execute(state)
    return {
        "winner": state["winner"],
        "explanation": state["explanation"],
        "used_full_tables": state["used_full_tables"],
    }
