"""Tier 3 - LLM verification of a candidate's curated table against source.

Modeled on (not copy-pasted from) the multi-agent mode's own verify/correct
loop (`extractor/agents/pk_pe_agents/pk_pe_verification_step.py`'s
`PKPE_VERIFICATION_SYSTEM_PROMPT` + `PKPEVerificationStepResult`), per the
explicit design decision to make this "very similar to [the] verification
agent in multi-agent mode" rather than inventing a new prompt from scratch.

Two deliberate departures from that prompt, both aimed at making error_rate
computable without regex-parsing free text:

- No "Previous Verification Attempts" section - this runs once, no
  correction loop, so that section would always render as "None" and the
  instructions for handling a non-empty one would never apply.
- The output schema (`CurateAgentVerificationResult` below) asks for
  structured fields - `error_rows: [{idx, column, found_value,
  expected_value}]`, `extra_rows: [idx, ...]`, `missing_rows: [str, ...]` -
  instead of one free-text blob in a fixed "idx N, Col ...: change ... to
  ..." format that then has to be regex-parsed back apart
  (`PKPEVerificationStepResult.explanation`). Structured fields also make
  the production step's own hallucination guards (dropping no-op changes,
  and changes whose claimed *current* value doesn't match the real curated
  table - see `PKPECuratedTablesVerificationStep._remove_noop_fixes`/
  `_remove_stale_fixes`) simple field comparisons instead of regex, and let
  `_count_errors` bound-check `idx` against the real curated table's row
  count directly - closing a gap the regex version had: a real run on PMID
  18426260 (94 rows) returned "idx 202" and nothing caught it, since the
  old `_remove_stale_fixes` only validated `idx < len(df)` and silently kept
  anything outside that range instead of dropping it.
"""

from __future__ import annotations

import json
import logging

from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from extractor.agents.agent_factory import get_agent_llm
from extractor.agents.pk_pe_agents.pk_pe_common_step import PKPECommonStep
from extractor.constants import COT_USER_INSTRUCTION

logger = logging.getLogger(__name__)


def _log_tier3_call(paper_title: str, domain: str, schema: str | None, curated_table: str,
                     res: "CurateAgentVerificationResult", token_usage: dict,
                     error_row_count: int, missed_row_count: int) -> None:
    """Log a real Tier 3 LLM call's full structured output - the
    error_rows/extra_rows/missing_rows/explanation that check_validity()'s
    own tier3 dict throws away (see _count_errors()'s docstring; it keeps
    only error_row_count/missed_row_count). Uses this module's own
    `logger` - extractor/log_utils.py's `initialize_logger()` already wires
    every "extractor.*" logger (by propagation from the "extractor" entry
    in its `log_entries`) to a file under $LOGS_FOLDER, the same mechanism
    app.py/app_script.py/app_script_pmids.py/benchmark/conftest.py already
    use - so this needs no logging setup of its own, only a caller that has
    called `initialize_logger()` (e.g. scripts/calibrate_validity_threshold.py)."""
    record = {
        "paper_title": paper_title,
        "domain": domain,
        "schema": schema,
        "curated_table": curated_table,
        "token_usage": token_usage,
        "llm_output": res.model_dump(),
        "counted": {"error_row_count": error_row_count, "missed_row_count": missed_row_count},
    }
    logger.info("TIER3_LLM_CALL %s", json.dumps(record))

# PKPE_VERIFICATION_SYSTEM_PROMPT (pk_pe_verification_step.py), minus its
# "Previous Verification Attempts" section, with the "Your Output"/"Error
# Listing Format" sections rewritten for CurateAgentVerificationResult's
# structured fields instead of one free-text blob - see module docstring.
CURATE_AGENT_VERIFICATION_PROMPT = """
You are a biomedical data verification assistant with expertise in {domain} and data accuracy validation.
Your task is to carefully examine the **source paper title and tables**, and determine whether the **curated {domain} data table** is an accurate and faithful representation of the information provided in the source.

---

### **Your Responsibilities**

* Verify that all values in the curated table exactly match or are correctly derived from the source table(s) in the paper.
* Check that the table structure (rows, columns, units, and headers) is curated correctly from the source.
* Identify any discrepancies in numerical values, missing data, wrong units, or incorrect associations (e.g., a value placed in the wrong row or column).
{completeness_rules}
* Consider the context from the paper title if needed (e.g., study type, drug, population) to interpret ambiguous values.

---

### **Input**

You will be given:

* **Paper Title**: The title of the publication.
* **Paper Abstract**: The abstract of the publication.
* **Source Table(s) or full text**: Table(s) extracted directly from the publication, preserving structure and labels, or the full text of the publication.
* **Curated Table**: The data table that has been curated from the above source for downstream use.

---

### **Your Output**

Respond using the exact JSON schema below:

```
{{
  "correct": <boolean, True / False>,
  "error_rows": [
    {{"idx": <int, 0-based row position>, "column": <string, exact column name>, "found_value": <string, EXACTLY what is currently in the curated table at this idx/column>, "expected_value": <string, what it should be per the source>}}
  ],
  "extra_rows": [<int, idx of a fabricated curated row with no source counterpart, that should be removed>],
  "missing_rows": [<string, one per source row with no corresponding row in the curated table at all, e.g. "Drug=\\"X\\", Value=\\"5\\"" or "Drug=\\"X\\", Case=\\"2\\", Value=\\"Not measured\\"" for a negative/null finding>],
  "explanation": <string, max 100 words, short human-readable summary. "All values match." if correct.>
}}
```

If the curated table is fully correct: `correct=true`, `error_rows`/`extra_rows`/`missing_rows` all empty, `explanation="All values match."`.

List **one entry per wrong cell** in `error_rows` - if a row has two wrong columns, that's two entries, both with the same `idx`.

---

### **Important Rules**

* The columns in the curated table are fixed — do NOT question column names or order.
* `idx` is the row's **0-based POSITION** in the curated table (first data row after the header = idx 0, second data row = idx 1, etc.), NOT a value from any column. For example, if the first data row has Patient ID "3", that row is still idx 0 (not idx 3).
* `idx` MUST refer to a row that actually exists in the curated table (`0 <= idx <` the curated table's real row count). **Never invent or guess an idx** - if you are not sure a row exists at that position, do not list it.
* `found_value` MUST be copied **exactly** as it currently appears in the curated table at that `idx`/column - do not paraphrase, round, or guess it. If you are not sure what the current value is, do not list that cell.
* Only include a cell in `error_rows` if `expected_value` is genuinely different from `found_value` - never list a no-op (e.g. do NOT list found="0.001", expected="0.001").
* Focus on **substantial mismatches** in values or structure. Minor typos, slight wording differences, or small formatting variations are acceptable.
  In particular,
    - **ignore** footnote markers or superscripts attached to values (e.g., "< LODa" vs "< LOD", "3.5*" vs "3.5").
    - **ignore** the difference between "nan" and "N/A" and "N/A" and "NA", and so on.
* When values in text and table disagree, treat the **table values as ground truth**.
* You MUST list **EVERY** incorrect value - do not give partial examples.
* You MUST list **EVERY** missing row, as defined by the completeness rules above - do not give partial examples.
* `explanation` must NOT include thinking process ("re-evaluate", "wait", "let's check", etc.) or restate the error list - just a short summary, max 100 words.

---

### **Input**

#### **Paper Title**

{paper_title}

#### **Paper Abstract**

{paper_abstract}

#### **Source Table(s) or full text**

{source_tables}

#### **Curated Table**

{curated_table}

---
"""


# What "missing row" means depends on what the curated table is supposed to
# hold. The per-subject rules were added after PMID 34746508 (pk_individual:
# 10 of 15 individual rows silently dropped, caught only once the prompt said
# to go subject by subject). Applied to pk_summary they backfire: on PMID
# 16143486 Tier 3 reported Table 4's 32 individual parturient measurements as
# "missing" from both arms, while the gold pk_summary table holds only that
# table's 4 mean rows - individual data is the pk_individual pipeline's job.
INDIVIDUAL_COMPLETENESS_RULES = """\
* **Check completeness, not just correctness that's already there.** A source table is often "wide": one row per measurement, one column per subject/case (e.g. Case 1 ... Case 5). Each subject's cell for that measurement is one data point that should have its own row in the curated (long-format) table - go column by column, subject by subject, and confirm every one of them made it in, not just the ones you can already see curated correctly.
* **A negative or null finding is still a real data point.** "Not measured", "Not detected", "ND", "BLQ", "N/A", "-", "none found", etc. in the source, reported for a specific subject/row, must still appear as a curated row (with that text as the value) - it is NOT evidence that nothing should be curated for that subject. A curated table that only keeps rows with a numeric result and silently drops every subject whose result was negative/null is missing those rows."""

SUMMARY_COMPLETENESS_RULES = """\
* **Check completeness at the level this table is curated at: summary statistics.** The curated table holds one row per reported summary value (mean, median, geometric mean, CI, range, SD, n, ...). Confirm every reported summary value in the source has a curated row. Per-subject / individual measurements in the source (e.g. one row per patient or case) are NOT expected in this table - do NOT list them as missing rows."""

# The full-text pipelines curate from prose, one row per distinct reported
# fact (a dosing regimen, a sampling scheme, a characteristic statistic); the
# rules say what "one row" is so the agent neither demands a row per sentence
# nor accepts one summary row for several regimens.
TEXT_COMPLETENESS_RULES = {
    "pk_drug_summary": """\
* **Completeness = one row per distinct dosing regimen** (drug or metabolite x dose amount x unit x frequency x schedule x route x population group) the paper describes for its own study subjects. A regimen stated in the text but absent from the curated table is a missing row. Doses of other studies cited in the discussion are NOT expected.""",
    "pk_drug_individual": """\
* **Completeness = one row per individual x dosing regimen** reported for identified patients / cases of this study (a case report's single patient counts). Group-level dosing with no individual identifier is NOT expected in this table.""",
    "pk_specimen_summary": """\
* **Completeness = one row per distinct sampling scheme** (specimen type x population group x sampling time / time unit, with the sample count and subject count when stated). Report a scheme the text describes that has no curated row as missing; do NOT expect rows for measurements or results.""",
    "pk_specimen_individual": """\
* **Completeness = one row per individual x specimen sampling** reported for identified patients / cases. Group-level sampling with no individual identifier is NOT expected.""",
    "pk_population_summary": """\
* **Completeness = one row per reported characteristic statistic per population group** (age, weight, sex counts, gestational age, ... with their mean / median / count, variation and interval). Characteristics the source reports with a value but which have no curated row are missing rows. PK parameters and drug concentrations are NOT population characteristics and are NOT expected here.""",
    "pk_population_individual": """\
* **Completeness = one row per individual x characteristic** (age, weight, sex, gestational age, ...) reported for identified patients / cases. Group statistics are NOT expected here.""",
    "pe_study_info": """\
* **This table has exactly ONE row** describing the study (type, design, data source, population, criteria, outcomes, subject count). Do not report additional rows as missing; a wrong or empty cell in the one row is an error_rows entry.""",
    "pe_study_outcome": """\
* **Completeness = one row per reported outcome statistic** (each numeric value with its exposure / outcome / characteristic context, statistic type, variation, interval and p value). Every number of the source table that is an outcome result should have a row; header counts and footnotes are NOT expected.""",
}


def completeness_rules_for(schema: str | None) -> str:
    """Per-subject rules for individual-level PK table schemas (and when the
    schema is unknown - the stricter, original behaviour); summary-level
    rules for pk_summary; per-schema rules for the full-text and PE schemas."""
    if schema in TEXT_COMPLETENESS_RULES:
        return TEXT_COMPLETENESS_RULES[schema]
    if schema is None or "individual" in schema:
        return INDIVIDUAL_COMPLETENESS_RULES
    return SUMMARY_COMPLETENESS_RULES


# --- Empty-table assessment -------------------------------------------------
#
# An empty candidate table has two possible meanings: the model failed, or the
# source table holds no data of the kind this schema curates, in which case
# empty is the right answer. The Tier 4 audit (docs/cascade_mode_results.md)
# found the cascade treating every empty table as a failure (error_rate 1.0)
# and losing twice out of three: PMID 18426260 table 2 is a "Mean ± SD
# (Range)" table with no patient identifiers - not pk_individual data - and
# PMID 34746508 table 3 lists cases from other publications, which the gold
# annotation does not curate. Arm A correctly returned nothing on both; the
# judge, told only "zero rows, error_rate 1.000", picked Arm B's fabricated
# rows. These rules state what counts as curatable for each schema so one LLM
# call can tell the two meanings apart. The literature-review exclusion is an
# annotation policy, not something a model infers, so it is spelled out.

INDIVIDUAL_APPLICABILITY_RULES = """\
* **Curatable data for this table** = drug concentration, PK-parameter or ratio values reported for **individual** subjects / patients / cases / mother-infant pairs, **measured in this study** - one row or one column per individual (e.g. "Patient 1 ... Patient 11", "Case 1 ... Case 5", "Subject A"), including per-individual null findings ("Not measured", "Not detected", "ND", "BLQ", "-").
* **NOT curatable - an empty table is CORRECT** when the source table holds only:
  - **group statistics** (mean ± SD, median [IQR], range, n (%), CI, geometric mean) with no per-individual values;
  - **cases or results from other published studies** (a citation column or author-year labels such as "Author et al. 2016" - those are not this paper's own measurements);
  - demographics / baseline characteristics; dosing schedules without measured values;
  - regression, covariate or population-PK model outputs."""

SUMMARY_APPLICABILITY_RULES = """\
* **Curatable data for this table** = **summary statistics** of drug concentrations or PK parameters (mean, median, geometric mean, SD, CV, SE, CI, range, IQR, n) for this study's own groups / cohorts / time points.
* A per-subject table that ALSO carries a mean / median / summary row has curatable data (those summary rows). A table with only per-subject raw values and no summary row has none for this table type.
* **NOT curatable - an empty table is CORRECT** when the source table holds only: demographics / baseline characteristics; regression, covariate or population-PK model outputs; results of other published studies; dosing schedules without measured values."""


TEXT_APPLICABILITY_RULES = {
    "pk_drug_summary": """\
* **Curatable data** = any dosing regimen of the study drug(s) given to this study's own subjects or groups: drug / metabolite name with a dose amount, unit, frequency, schedule or route (e.g. "150 mg orally twice daily", "a single 2 g IV dose").
* **NOT curatable - an empty table is CORRECT** when the source mentions no dosing at all for this study's subjects (e.g. a purely analytical-method paper, a modelling paper reusing published data, or a study of endogenous substances with no drug given).""",
    "pk_drug_individual": """\
* **Curatable data** = dosing regimens reported for identified individuals (case 1, patient A, the mother, the infant ...) of this study: drug / metabolite with dose amount, unit, frequency, schedule or route per individual.
* **NOT curatable - an empty table is CORRECT** when dosing is reported only at group level (a cohort given the same regimen with no per-patient dosing) or not at all.""",
    "pk_specimen_summary": """\
* **Curatable data** = how biological specimens were collected for this study's groups: specimen type (blood, plasma, urine, breast milk, cord blood, ...) with sample counts, subject counts, sampling times or time units.
* **NOT curatable - an empty table is CORRECT** when the source describes no specimen collection of its own (e.g. a review, a modelling paper, a survey with no biological sampling).""",
    "pk_specimen_individual": """\
* **Curatable data** = specimen sampling reported for identified individuals (case 1, patient A ...) of this study: specimen type with sample count or sampling time per individual.
* **NOT curatable - an empty table is CORRECT** when sampling is described only at group level or not at all.""",
    "pk_population_summary": """\
* **Curatable data** = characteristics of this study's population groups with a reported statistic: age, weight, height, BMI, sex counts, gestational age, race, comorbidities, ... as a mean / median / count / percentage, with or without SD, range, IQR or CI.
* **NOT curatable - an empty table is CORRECT** when the source reports no population characteristics (e.g. an in-vitro or purely analytical paper) or only PK parameters and drug concentrations, which are NOT population characteristics.""",
    "pk_population_individual": """\
* **Curatable data** = characteristics (age, weight, sex, gestational age, ...) reported for identified individuals (case 1, patient A, the mother, the infant ...) of this study.
* **NOT curatable - an empty table is CORRECT** when characteristics are reported only as group statistics or not at all.""",
    "pe_study_info": """\
* **Curatable data** = the study itself: every paper that reports a study has a study type, design, data source, population, criteria or outcomes - this table is almost never legitimately empty.
* **NOT curatable - an empty table is CORRECT** only when the source is not a study at all (an editorial, a letter with no study, a methods note).""",
    "pe_study_outcome": """\
* **Curatable data** = numeric outcome results of this study: counts, percentages, means, rates, odds / risk ratios, with their exposure / outcome context, variation, interval or p value.
* **NOT curatable - an empty table is CORRECT** when the source table holds no outcome results: e.g. only baseline / demographic characteristics, only definitions or methods, or only data cited from other studies.""",
}


def applicability_rules_for(schema: str | None) -> str:
    if schema in TEXT_APPLICABILITY_RULES:
        return TEXT_APPLICABILITY_RULES[schema]
    if schema is None or "individual" in schema:
        return INDIVIDUAL_APPLICABILITY_RULES
    return SUMMARY_APPLICABILITY_RULES


_SCHEMA_LABELS = {
    "pk_individual": "individual-level pharmacokinetic table (one row per measurement of one individual)",
    "pk_summary": "summary-level pharmacokinetic table (one row per reported summary statistic)",
    "pk_drug_summary": "study-level drug dosing table (one row per dosing regimen per population group)",
    "pk_drug_individual": "individual-level drug dosing table (one row per individual per dosing regimen)",
    "pk_specimen_summary": "study-level specimen sampling table (one row per specimen type per population group per sampling scheme)",
    "pk_specimen_individual": "individual-level specimen sampling table (one row per individual per specimen)",
    "pk_population_summary": "population characteristics table (one row per characteristic statistic per group)",
    "pk_population_individual": "individual characteristics table (one row per individual per characteristic)",
    "pe_study_info": "pharmacoepidemiology study design table (exactly one row describing the study)",
    "pe_study_outcome": "pharmacoepidemiology outcome table (one row per reported outcome statistic)",
}

CURATE_AGENT_EMPTY_TABLE_PROMPT = """
You are a biomedical data verification assistant with expertise in {domain} and data curation.
A curation step that builds a **{schema_label}** from ONE source of a publication (a table, or the paper's full text for the full-text table types) returned an **empty table (zero rows)**.
Decide which of two things happened:

* **Correct**: the source contains **no data of the kind this table type curates**, so an empty table is the right output.
* **Failure**: the source **does** contain such data, and rows should have been curated.

---

### **What counts as curatable data for this table type**

{applicability_rules}

---

### **Your Output**

Respond using the exact JSON schema below:

```
{{
  "has_curatable_data": <boolean, True if the source table contains data this table type should curate, False if an empty table is the correct output>,
  "expected_rows": [<string, one short description per row that SHOULD have been curated, e.g. "Case 2, cord blood, 4.1 μg/ml"; empty list when has_curatable_data is false. List EVERY such row, not examples.>],
  "explanation": <string, max 100 words, cite the feature of the source table that decided it, e.g. "rows are group means with no patient identifiers" or "rows are cases from cited studies (Author et al. 2016)" or "one column per patient, values are individual concentrations">
}}
```

---

### **Input**

#### **Paper Title**

{paper_title}

#### **Paper Abstract**

{paper_abstract}

#### **Source Table or Text**

{source_tables}

---
"""


class EmptyTableAssessment(BaseModel):
    has_curatable_data: bool = Field(
        description="True if the source table contains data this table type should curate; False if an empty table is correct"
    )
    expected_rows: list[str] = Field(
        default_factory=list,
        description="One short description per row that should have been curated; empty when has_curatable_data is false",
    )
    explanation: str = Field(description="Short (<=100 words) reason, citing the source-table feature that decided it")


class CurateAgentEmptyTableStep(PKPECommonStep):
    """Single-pass check of an EMPTY candidate table: is the source table even
    curatable for this schema? Runs only inside the cascade's ValidityChecker,
    on the zero-row path (`validity_checker.check_validity`)."""

    def __init__(self, llm: BaseChatOpenAI, domain: str = "pharmacokinetics",
                 schema: str | None = None):
        super().__init__(llm)
        self.step_name = "Curate Agent Empty Table Assessment Step"
        self.domain = domain
        self.schema = schema

    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        agent = self.get_agent(llm=self.llm)
        prompt = CURATE_AGENT_EMPTY_TABLE_PROMPT.format(
            domain=self.domain,
            schema_label=_SCHEMA_LABELS.get(self.schema, self.schema or "pharmacokinetic data table"),
            applicability_rules=applicability_rules_for(self.schema),
            paper_title=state["paper_title"],
            paper_abstract=state["paper_abstract"],
            source_tables=state["source_tables"],
        )
        res, _, token_usage, reasoning = agent.go(
            system_prompt=prompt,
            instruction_prompt=COT_USER_INSTRUCTION,
            schema=EmptyTableAssessment,
        )
        self._print_step(state, step_output=reasoning or "N/A")
        logger.info("TIER3_EMPTY_TABLE_CALL %s", json.dumps({
            "paper_title": state["paper_title"],
            "domain": self.domain,
            "schema": self.schema,
            "source_tables": state["source_tables"],
            "token_usage": token_usage,
            "llm_output": res.model_dump(),
        }))
        state["has_curatable_data"] = bool(res.has_curatable_data)
        state["expected_rows"] = list(res.expected_rows or []) if res.has_curatable_data else []
        state["explanation"] = res.explanation
        return state, token_usage


def assess_empty_table(paper_title, paper_abstract, source_tables, schema=None,
                       llm=None, domain="pharmacokinetics", step_output_callback=None) -> dict:
    """One LLM call: does `source_tables` hold data `schema` should curate?

    Used by `validity_checker.check_validity` when a candidate table has zero
    rows. Returns {"has_curatable_data": bool, "expected_rows": [str],
    "explanation": str}. `llm` defaults to `get_agent_llm()`, like `verify`.
    """
    step = CurateAgentEmptyTableStep(llm=llm or get_agent_llm(), domain=domain, schema=schema)
    state = {
        "step_output_callback": step_output_callback,
        "paper_title": paper_title,
        "paper_abstract": paper_abstract,
        "source_tables": source_tables,
    }
    state = step.execute(state)
    return {
        "has_curatable_data": state["has_curatable_data"],
        "expected_rows": state["expected_rows"],
        "explanation": state["explanation"],
    }


class ErrorEntry(BaseModel):
    idx: int = Field(description="0-based row position in the curated table (NOT a value from any column)")
    column: str = Field(description="Exact column name with the wrong value")
    found_value: str = Field(description="The value currently in the curated table at this idx/column, copied exactly")
    expected_value: str = Field(description="What the value should be, per the source")


class CurateAgentVerificationResult(BaseModel):
    correct: bool = Field(description="Whether the curated table is accurate and faithful to the source")
    error_rows: list[ErrorEntry] = Field(default_factory=list, description="One entry per wrong cell; empty if correct")
    extra_rows: list[int] = Field(default_factory=list, description="idx of fabricated curated rows with no source counterpart, to be removed")
    missing_rows: list[str] = Field(default_factory=list, description="One short description per source row missing from the curated table entirely")
    explanation: str = Field(description="Short (<=100 words) human-readable summary. 'All values match.' if correct.")


def _count_errors(res: CurateAgentVerificationResult, curated_table: str) -> tuple[int, int, str]:
    """(error_row_count, missed_row_count, explanation) from a structured
    CurateAgentVerificationResult, after dropping entries this module can
    prove are wrong on their own terms - same intent as the production
    verification step's `_remove_noop_fixes`/`_remove_stale_fixes`, but as
    plain field comparisons instead of regex over free text:

    - no-op: `found_value == expected_value` (claimed "fix" changes nothing).
    - out-of-range: `idx` outside the real curated table's row range -
      the model inventing a row that doesn't exist.
    - stale: `found_value` doesn't match what's actually at that idx/column
      in the curated table - the model's claim about the *current* state
      is simply false, independent of whether its *fix* is right.

    `extra_rows` gets the same out-of-range check (an out-of-range idx can't
    be "removed"). `missing_rows` isn't filtered - there's no idx to
    fact-check against the curated table for a row that isn't in it.

    Distinct `idx` values across the kept `error_rows` and `extra_rows`
    count once each (a row with 2 wrong columns is 1 error row, matching
    the production step's own per-row, not per-cell, counting).
    """
    df = None
    try:
        from TabFuncFlow.utils.table_utils import markdown_to_dataframe
        df = markdown_to_dataframe(curated_table)
    except Exception:
        logger.warning("Could not parse curated_table for Tier 3 stale-fix check", exc_info=True)
    total_rows = len(df) if df is not None else None

    kept_errors = []
    for e in res.error_rows:
        if str(e.found_value).strip() == str(e.expected_value).strip():
            continue
        if total_rows is not None and not (0 <= e.idx < total_rows):
            continue
        if df is not None and e.column in df.columns and e.idx < len(df):
            actual = str(df.iloc[e.idx][e.column]).strip()
            if actual != str(e.found_value).strip():
                continue
        kept_errors.append(e)

    kept_extra = [idx for idx in res.extra_rows if total_rows is None or 0 <= idx < total_rows]

    error_idxs = {e.idx for e in kept_errors} | set(kept_extra)
    missed_row_count = len(res.missing_rows)

    if not error_idxs and not missed_row_count:
        return 0, 0, "All values match."

    lines = [f'idx {e.idx}, Col "{e.column}": change "{e.found_value}" to "{e.expected_value}"' for e in kept_errors]
    lines += [f"Extra idx {idx}: should be removed" for idx in kept_extra]
    lines += [f"Missing row: {m}" for m in res.missing_rows]
    return len(error_idxs), missed_row_count, "\n".join(lines)


class CurateAgentVerificationStep(PKPECommonStep):
    """Single-pass Tier 3 check. Not part of PKPEManager's own verify/correct
    loop - this runs only inside the CurateAgent cascade's ValidityChecker."""

    def __init__(self, llm: BaseChatOpenAI, domain: str = "pharmacokinetics",
                 schema: str | None = None):
        super().__init__(llm)
        self.step_name = "Curate Agent LLM Verification Step"
        self.domain = domain
        self.schema = schema

    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        agent = self.get_agent(llm=self.llm)
        prompt = CURATE_AGENT_VERIFICATION_PROMPT.format(
            domain=self.domain,
            completeness_rules=completeness_rules_for(self.schema),
            paper_title=state["paper_title"],
            paper_abstract=state["paper_abstract"],
            source_tables=state["source_tables"],
            curated_table=state["curated_table"],
        )
        res, _, token_usage, reasoning = agent.go(
            system_prompt=prompt,
            instruction_prompt=COT_USER_INSTRUCTION,
            schema=CurateAgentVerificationResult,
        )
        self._print_step(state, step_output=reasoning or "N/A")

        error_row_count, missed_row_count, explanation = _count_errors(res, state["curated_table"])
        _log_tier3_call(
            paper_title=state["paper_title"], domain=self.domain, schema=self.schema,
            curated_table=state["curated_table"], res=res, token_usage=token_usage,
            error_row_count=error_row_count, missed_row_count=missed_row_count,
        )
        state["correct"] = error_row_count == 0 and missed_row_count == 0
        state["error_row_count"] = error_row_count
        state["missed_row_count"] = missed_row_count
        state["explanation"] = explanation
        return state, token_usage


def verify(paper_title, paper_abstract, source_tables, curated_table,
           llm=None, domain="pharmacokinetics", schema=None,
           step_output_callback=None) -> dict:
    """Ask the verification agent to check one curated table against its
    source, and count error/missed rows from its structured response.

    `schema` picks the completeness rules (see completeness_rules_for);
    None keeps the per-subject rules. Returns {"correct": bool,
    "error_row_count": int, "missed_row_count": int, "explanation": str}.
    `llm` defaults to `get_agent_llm()` (the AGENT_LLM env var), same as the
    rest of the PKPE agent machinery.
    """
    step = CurateAgentVerificationStep(llm=llm or get_agent_llm(), domain=domain, schema=schema)
    state = {
        "step_output_callback": step_output_callback,
        "paper_title": paper_title,
        "paper_abstract": paper_abstract,
        "source_tables": source_tables,
        "curated_table": curated_table,
    }
    state = step.execute(state)
    return {
        "correct": state["correct"],
        "error_row_count": state["error_row_count"],
        "missed_row_count": state["missed_row_count"],
        "explanation": state["explanation"],
    }
