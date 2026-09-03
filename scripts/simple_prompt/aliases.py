"""Vocabulary equivalences for scoring pk-individual output.

Only used when the scorer is run with --aliases. Kept separate and explicit
because these are judgement calls about what counts as "the same answer", and
they should be argued about in review rather than buried in the scorer.

Pure formatting differences (micro sign, case, whitespace) are NOT here - the
scorer always normalises those, since they carry no meaning.
"""

# Safe: the two strings denote the same thing. The gold's wording differs from
# the controlled vocabulary the legacy extractor (and the skill) emits.
SAFE = {
    "Pregnancy stage": {
        "delivery": ["parturition/labor/delivery", "parturition", "labor", "birth",
                     "at delivery"],
        "lactation": ["nursing/breastfeeding/lactation", "breastfeeding", "nursing"],
        "postpartum": ["post partum", "post-partum"],
        "1st trimester": ["trimester 1", "first trimester"],
        "2nd trimester": ["trimester 2", "second trimester", "2nd trimster"],
        "3rd trimester": ["trimester 3", "third trimester"],
        "pregnancy": ["pregnant", "gestation", "antenatal"],
    },
    "Pediatric/Gestational age": {
        "maternal": ["mother", "mothers"],
        "infant": ["infants", "neonate", "neonates", "newborn"],
        "fetus": ["fetal", "foetus"],
        "pediatric": ["paediatric", "child", "children"],
    },
    "Specimen": {
        "breast milk": ["milk", "human milk", "breastmilk"],
        "umbilical cord blood": ["cord blood", "umbilical blood", "cord"],
        "plasma": ["blood plasma"],
        "serum": ["blood serum"],
    },
    "Parameter type": {
        "average concentration": ["cavg", "c avg", "mean concentration"],
        "m/p auc": ["m/p auc ratio", "milk/plasma auc ratio", "milk to plasma auc"],
        "concentration": ["level", "plasma level", "serum level", "conc"],
    },
}

# Debatable: arguably the same subject, but the two vocabularies are not
# strictly synonymous. Review these before trusting a score that uses them.
DEBATABLE = {
    "Population": {
        "maternal": ["adults", "adult", "adult female", "pregnant women", "mothers",
                     "women"],
        "pediatric": ["infants", "infant", "neonate", "neonates", "newborn",
                      "children", "child"],
    },
}


def build(include_debatable: bool = True):
    """Flatten to {column: {variant: canonical}}."""
    out = {}
    sources = [SAFE] + ([DEBATABLE] if include_debatable else [])
    for src in sources:
        for col, groups in src.items():
            table = out.setdefault(col, {})
            for canon, variants in groups.items():
                table[canon] = canon
                for v in variants:
                    table[v] = canon
    return out
