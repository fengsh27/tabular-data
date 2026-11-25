"""Column schemas of the eight curation pipelines that share one generic
simple-prompt parser (`_common_generic.SchemaParser`).

pk_individual and pk_summary keep their own hand-tuned modules (`_common.py`,
`_common_summary.py`); everything else is described here as data and parsed
by the same code. Column names and order are the ones the production
LangGraph workflows emit after their final rename (`go_full_text` /
`go_md_table` in extractor/agents/<pipeline>/*_workflow.py) - i.e. what
`benchmark/data/pk-pe/<version>/{pmid}_{pipeline}_{model}.csv` files hold -
so a simple-prompt table, an Arm B table and a benchmark file all line up
column for column.

Each spec also says what the pipeline's *unit of work* is for the cascade
(`unit_kind`): the two PK table pipelines and pe_study_outcome curate one
selected table per call; pk_drug_*, pk_specimen_* and pe_study_info read the
paper's full text (one unit per paper); pk_population_* use the demographic
tables when the selector finds any and the full text otherwise - exactly the
input each production AgentTool hands its workflow
(extractor/agents/pk_pe_agents/pk_pe_agent_tools.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

UNIT_TABLE = "table"            # one selected source table per unit
UNIT_TEXT = "text"              # the paper's full text, one unit per paper
UNIT_TABLE_OR_TEXT = "table_or_text"  # selected tables if any, else the full text

# The population / pregnancy categories the production refine prompts ask for.
# Used only as realignment tie-breakers by the parser (never to validate or
# rewrite a cell) - see _common_summary._VOCAB.
_POPULATION_VOCAB = {"nonpregnant", "maternal", "pediatric", "adults", "adult",
                     "maternal/pediatric", "healthy adults", "pregnant women"}
_PREGNANCY_VOCAB = {"pre-pregnancy", "trimester 1", "trimester 2", "trimester 3", "fetus",
                    "parturition", "labor", "delivery", "postpartum", "nursing",
                    "breastfeeding", "lactation", "pregnancy"}
_DEMOGRAPHIC_ALIASES = {
    "population": "Population",
    "pregnancystage": "Pregnancy stage",
    "pediatricgestationalage": "Pediatric/Gestational age",
    "gestationalage": "Pediatric/Gestational age",
    "pediatricage": "Pediatric/Gestational age",
}
_NOTE_ALIASES = {"note": "Note", "notes": "Note", "sourcetext": "Note", "source": "Note"}


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    cols: tuple[str, ...]
    aliases: dict[str, str]                 # squashed header spelling -> canonical column
    numeric_cols: frozenset[str]            # a cell here must start with a number (or be blank)
    text_cols: frozenset[str]               # a cell here must NOT be a bare number
    required_cols: tuple[str, ...]          # every row needs these (blank = the row slipped)
    unit_kind: str
    vocab: dict[str, frozenset[str]] = field(default_factory=dict)
    min_header_hits: int = 4                # recognised names needed to call a line the header

    def __post_init__(self):
        assert self.unit_kind in (UNIT_TABLE, UNIT_TEXT, UNIT_TABLE_OR_TEXT), self.unit_kind
        for group in (self.numeric_cols, self.text_cols, self.required_cols, self.vocab):
            for c in group:
                assert c in self.cols, f"{self.name}: {c!r} is not a column"
        for canonical in self.aliases.values():
            assert canonical in self.cols, f"{self.name}: alias target {canonical!r} is not a column"


def _spec(name, cols, aliases, numeric, text, required, unit_kind, vocab=None, min_header_hits=4):
    # every canonical column is accepted under its own squashed spelling too
    squashed = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in cols}
    return SchemaSpec(
        name=name, cols=tuple(cols), aliases={**squashed, **aliases},
        numeric_cols=frozenset(numeric), text_cols=frozenset(text), required_cols=tuple(required),
        unit_kind=unit_kind, vocab={k: frozenset(v) for k, v in (vocab or {}).items()},
        min_header_hits=min_header_hits,
    )


_DRUG_COLS = ["Drug/Metabolite name", "Dose amount", "Dose unit", "Dose frequency",
              "Dose schedule", "Dose route", "Population", "Pregnancy stage",
              "Pediatric/Gestational age"]
_DRUG_ALIASES = {
    "drugmetabolitename": "Drug/Metabolite name", "drugname": "Drug/Metabolite name",
    "drug": "Drug/Metabolite name", "metabolite": "Drug/Metabolite name",
    "doseamount": "Dose amount", "dose": "Dose amount", "amount": "Dose amount",
    "doseunit": "Dose unit", "unit": "Dose unit",
    "dosefrequency": "Dose frequency", "frequency": "Dose frequency",
    "doseschedule": "Dose schedule", "schedule": "Dose schedule",
    "doseroute": "Dose route", "route": "Dose route",
    **_DEMOGRAPHIC_ALIASES, **_NOTE_ALIASES,
}

_SPECIMEN_ALIASES = {
    "specimen": "Specimen", "specimentype": "Specimen", "matrix": "Specimen",
    "samplen": "Sample N", "numberofsamples": "Sample N", "samples": "Sample N",
    "sampletime": "Sample time", "samplingtime": "Sample time", "time": "Sample time",
    "timeunit": "Time unit", "unit": "Time unit",
    **_DEMOGRAPHIC_ALIASES, **_NOTE_ALIASES,
}

_POPULATION_ALIASES = {
    "characteristic": "Characteristic", "populationcharacteristic": "Characteristic",
    "patientcharacteristic": "Characteristic",
    "characteristicsubcategory": "Characteristic subcategory", "subcategory": "Characteristic subcategory",
    "characteristicunit": "Characteristic unit", "unit": "Characteristic unit",
    "characteristicvalue": "Characteristic value", "mainvalue": "Characteristic value",
    "value": "Characteristic value",
    **_DEMOGRAPHIC_ALIASES, **_NOTE_ALIASES,
}

SPECS: dict[str, SchemaSpec] = {
    "pk_drug_summary": _spec(
        "pk_drug_summary",
        _DRUG_COLS + ["Subject N", "Note"],
        {**_DRUG_ALIASES, "subjectn": "Subject N", "populationn": "Subject N", "n": "Subject N"},
        numeric={"Dose amount", "Subject N"},
        text={"Drug/Metabolite name", "Dose unit", "Dose schedule", "Dose route"},
        required=["Drug/Metabolite name"], unit_kind=UNIT_TEXT,
        vocab={"Population": _POPULATION_VOCAB, "Pregnancy stage": _PREGNANCY_VOCAB},
    ),
    "pk_drug_individual": _spec(
        "pk_drug_individual",
        ["Patient ID"] + _DRUG_COLS + ["Note"],
        {**_DRUG_ALIASES, "patientid": "Patient ID", "subjectid": "Patient ID", "patient": "Patient ID",
         "caseid": "Patient ID", "case": "Patient ID"},
        numeric={"Dose amount"},
        text={"Drug/Metabolite name", "Dose unit", "Dose schedule", "Dose route"},
        required=["Patient ID", "Drug/Metabolite name"], unit_kind=UNIT_TEXT,
        vocab={"Population": _POPULATION_VOCAB, "Pregnancy stage": _PREGNANCY_VOCAB},
    ),
    "pk_specimen_summary": _spec(
        "pk_specimen_summary",
        ["Specimen", "Sample N", "Population", "Pregnancy stage", "Pediatric/Gestational age",
         "Subject N", "Sample time", "Time unit", "Note"],
        {**_SPECIMEN_ALIASES, "subjectn": "Subject N", "populationn": "Subject N", "n": "Subject N"},
        numeric={"Sample N", "Subject N"},
        text={"Specimen", "Time unit"},
        required=["Specimen"], unit_kind=UNIT_TEXT,
        vocab={"Population": _POPULATION_VOCAB, "Pregnancy stage": _PREGNANCY_VOCAB},
    ),
    "pk_specimen_individual": _spec(
        "pk_specimen_individual",
        ["Patient ID", "Specimen", "Sample N", "Population", "Pregnancy stage",
         "Pediatric/Gestational age", "Sample time", "Time unit", "Note"],
        {**_SPECIMEN_ALIASES, "patientid": "Patient ID", "subjectid": "Patient ID",
         "patient": "Patient ID", "caseid": "Patient ID", "case": "Patient ID"},
        numeric={"Sample N"},
        text={"Specimen", "Time unit"},
        required=["Patient ID", "Specimen"], unit_kind=UNIT_TEXT,
        vocab={"Population": _POPULATION_VOCAB, "Pregnancy stage": _PREGNANCY_VOCAB},
    ),
    "pk_population_summary": _spec(
        "pk_population_summary",
        ["Characteristic", "Characteristic subcategory", "Characteristic unit", "Characteristic value",
         "Statistics type", "Variation type", "Variation value", "Interval type", "Lower bound",
         "Upper bound", "Population", "Pregnancy stage", "Pediatric/Gestational age", "Subject N", "Note"],
        {**_POPULATION_ALIASES,
         "statisticstype": "Statistics type", "statistic": "Statistics type",
         "summarystatistics": "Statistics type", "statistics": "Statistics type",
         "variationtype": "Variation type", "variationvalue": "Variation value",
         "intervaltype": "Interval type",
         "lowerbound": "Lower bound", "lowerlimit": "Lower bound",
         "upperbound": "Upper bound", "upperlimit": "Upper bound", "highlimit": "Upper bound",
         "subjectn": "Subject N", "populationn": "Subject N", "n": "Subject N"},
        numeric={"Characteristic value", "Variation value", "Lower bound", "Upper bound", "Subject N"},
        text={"Characteristic", "Characteristic unit", "Statistics type", "Variation type", "Interval type"},
        required=["Characteristic"], unit_kind=UNIT_TABLE_OR_TEXT,
        vocab={"Population": _POPULATION_VOCAB, "Pregnancy stage": _PREGNANCY_VOCAB},
    ),
    "pk_population_individual": _spec(
        "pk_population_individual",
        ["Patient ID", "Characteristic", "Characteristic subcategory", "Characteristic unit",
         "Characteristic value", "Population", "Pregnancy stage", "Pediatric/Gestational age", "Note"],
        {**_POPULATION_ALIASES, "patientid": "Patient ID", "subjectid": "Patient ID",
         "patient": "Patient ID", "caseid": "Patient ID", "case": "Patient ID"},
        numeric={"Characteristic value"},
        text={"Characteristic", "Characteristic unit"},
        required=["Patient ID", "Characteristic"], unit_kind=UNIT_TABLE_OR_TEXT,
        vocab={"Population": _POPULATION_VOCAB, "Pregnancy stage": _PREGNANCY_VOCAB},
    ),
    "pe_study_info": _spec(
        "pe_study_info",
        ["Study type", "Population", "Study design", "Pregnancy stage", "Drug name", "Data source",
         "Inclusion criteria", "Exclusion criteria", "Outcomes", "Subject N"],
        {"studytype": "Study type", "type": "Study type",
         "studydesign": "Study design", "design": "Study design",
         "drugname": "Drug name", "drug": "Drug name", "drugs": "Drug name",
         "datasource": "Data source", "source": "Data source",
         "inclusioncriteria": "Inclusion criteria", "inclusion": "Inclusion criteria",
         "exclusioncriteria": "Exclusion criteria", "exclusion": "Exclusion criteria",
         "outcomes": "Outcomes", "outcome": "Outcomes",
         "subjectn": "Subject N", "n": "Subject N", "samplesize": "Subject N",
         "population": "Population", "pregnancystage": "Pregnancy stage"},
        numeric={"Subject N"},
        text={"Study type", "Study design", "Data source"},
        required=["Study type"], unit_kind=UNIT_TEXT,
        vocab={"Population": _POPULATION_VOCAB, "Pregnancy stage": _PREGNANCY_VOCAB},
    ),
    "pe_study_outcome": _spec(
        "pe_study_outcome",
        ["Characteristic", "Exposure", "Outcome", "Parameter unit", "Parameter statistic",
         "Parameter value", "Variation type", "Variation value", "Interval type", "Lower bound",
         "Upper bound", "P value"],
        {"characteristic": "Characteristic", "characteristicriskfactor": "Characteristic",
         "riskfactor": "Characteristic",
         "exposure": "Exposure",
         "outcome": "Outcome", "outcomes": "Outcome",
         "parameterunit": "Parameter unit", "unit": "Parameter unit", "mainvalueunit": "Parameter unit",
         "parameterstatistic": "Parameter statistic", "statistic": "Parameter statistic",
         "statisticstype": "Parameter statistic", "summarystatistics": "Parameter statistic",
         "parametervalue": "Parameter value", "value": "Parameter value", "mainvalue": "Parameter value",
         "variationtype": "Variation type", "variabilitystatistic": "Variation type",
         "variationvalue": "Variation value", "variabilityvalue": "Variation value",
         "intervaltype": "Interval type",
         "lowerbound": "Lower bound", "lowerlimit": "Lower bound", "intervallowbound": "Lower bound",
         "upperbound": "Upper bound", "upperlimit": "Upper bound", "highlimit": "Upper bound",
         "intervalhighbound": "Upper bound",
         "pvalue": "P value"},
        numeric={"Parameter value", "Variation value", "Lower bound", "Upper bound", "P value"},
        text={"Parameter unit", "Parameter statistic", "Variation type", "Interval type"},
        required=["Outcome"], unit_kind=UNIT_TABLE,
    ),
}

# The unit kind of every cascade schema, including the two hand-written ones.
UNIT_KINDS: dict[str, str] = {
    "pk_summary": UNIT_TABLE,
    "pk_individual": UNIT_TABLE,
    **{name: spec.unit_kind for name, spec in SPECS.items()},
}

ALL_SCHEMAS: tuple[str, ...] = tuple(UNIT_KINDS)
