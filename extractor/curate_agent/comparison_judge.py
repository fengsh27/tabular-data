"""Tier 4 - LLM semantic tie-break between two candidate curated tables.

Per docs/VERIFICATION_AGENT_DESIGN.md's ComparisonJudge: only invoked when
Tiers 1-2 leave the pick genuinely undecided (both candidates' `error_rate`
tied, including both-zero) - never as the primary mechanism, so the usual
unreliability of an LLM-judge stays confined to the narrow case it's
unavoidable for.

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

TIER4_REPORTS_ONLY_PROMPT = """
You are a biomedical data verification assistant with expertise in {domain}.
Two independent pipelines each curated the same source paper into a data
table (Candidate A and Candidate B). A deterministic validity check already
ran on both and could not tell them apart - their error rates are tied. You
are being asked to judge which candidate is actually more accurate, using
only the paper context and the two validity reports below - the actual
curated tables are NOT included in this pass, to save cost.

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
Two independent pipelines each curated the same source paper into a data
table (Candidate A and Candidate B). A deterministic validity check already
ran on both and could not tell them apart - their error rates are tied, and
a first LLM pass using only the validity reports could not disambiguate them
either. Judge which candidate is a more accurate, faithful curation of the
source table(s) below.

---

### **Paper Title**

{paper_title}

### **Paper Abstract**

{paper_abstract}

### **Source Table(s)**

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
    """Human-readable summary of a ValidityReport, for the prompt."""
    tier1 = report.get("tier1") or {}
    tier2 = report.get("tier2")
    lines = [
        f"- total_rows: {tier1.get('total_rows')}",
        f"- dropped/implausible rows: {tier1.get('dropped_rows')}",
        f"- avg per-row fit score: {tier1.get('avg_fit_score'):.2f}"
        if tier1.get("avg_fit_score") is not None else "- avg per-row fit score: N/A",
    ]
    if tier2 is None:
        lines.append("- provenance check: skipped (zero-row table)")
    else:
        lines.append(
            f"- provenance check: {tier2.get('unsupported')} misattributed "
            f"out of {tier2.get('checked_numbers')} numbers checked"
        )
    lines.append(f"- error_rate: {report.get('error_rate'):.3f}")
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

        reports_prompt = TIER4_REPORTS_ONLY_PROMPT.format(
            domain=self.domain,
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

        full_prompt = TIER4_FULL_PROMPT.format(
            domain=self.domain,
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
            step_output_callback=None) -> dict:
    """Judge which of two tied candidates is more accurate.

    Returns {"winner": "A" | "B", "explanation": str, "used_full_tables": bool}.
    `llm` defaults to `get_agent_llm()` (the AGENT_LLM env var), same as the
    rest of the PKPE agent machinery.
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
    }
    state = step.execute(state)
    return {
        "winner": state["winner"],
        "explanation": state["explanation"],
        "used_full_tables": state["used_full_tables"],
    }
