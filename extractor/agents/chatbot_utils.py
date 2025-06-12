from extractor.constants import (
    PROMPTS_NAME_PK_SUM,
    PROMPTS_NAME_PK_IND,
    PROMPTS_NAME_PK_SPEC_SUM,
    PROMPTS_NAME_PK_DRUG_SUM,
    PROMPTS_NAME_PK_POPU_SUM,
    PROMPTS_NAME_PK_SPEC_IND,
    PROMPTS_NAME_PK_DRUG_IND,
    PROMPTS_NAME_PK_POPU_IND,
    PROMPTS_NAME_PE_STUDY_INFO,
    PROMPTS_NAME_PE_STUDY_OUT,
)

try:
    from version import __version__
except Exception:
    __version__ = "unknown"

SYSTEM_NAME = f"Tabby (Tabular Curation Tool {__version__})"

INTRODUCTION = f"""You are {SYSTEM_NAME}, an intelligent assistant designed to extract and organize information from scientific literature into structured tables."""

REGULATION = """
Please follow these guidelines:
- Only answer questions that are directly related to the article and your assigned task.
- Do not speculate or provide information beyond the provided content or your assigned task.
- Keep your responses concise and evidence-based.
"""

COLUMN_DEFINITIONS = {
    PROMPTS_NAME_PK_SUM: "'Drug name', 'Analyte', 'Specimen', 'Population', 'Pregnancy stage', 'Pediatric/Gestational age', 'Subject N', 'Parameter type', 'Parameter unit', 'Parameter statistic', 'Parameter value', 'Variation type', 'Variation value', 'Interval type', 'Lower bound', 'Upper bound', 'P value', 'Time value', 'Time unit'",
    PROMPTS_NAME_PK_IND: "'Patient ID', 'Drug name', 'Analyte', 'Specimen', 'Population', 'Pregnancy stage', 'Pediatric/Gestational age', 'Parameter type', 'Parameter unit', 'Parameter value', 'Time value', 'Time unit'",
    PROMPTS_NAME_PK_SPEC_SUM: "'Specimen', 'Sample N', 'Population', 'Pregnancy stage', 'Pediatric/Gestational age', 'Subject N', 'Sample time', 'Time unit', 'Note'",
    PROMPTS_NAME_PK_DRUG_SUM: "'Drug/Metabolite name', 'Dose amount', 'Dose unit', 'Dose frequency', 'Dose schedule', 'Dose route', 'Population', 'Pregnancy stage', 'Pediatric/Gestational age', 'Subject N', 'Note'",
    PROMPTS_NAME_PK_POPU_SUM: "'Characteristic', 'Characteristic subcategory', 'Characteristic unit', 'Characteristic value', 'Statistics type', 'Variation type', 'Variation value', 'Interval type', 'Lower bound', 'Upper bound', 'Population', 'Pregnancy stage', 'Pediatric/Gestational age', 'Subject N', 'Note'",
    PROMPTS_NAME_PK_SPEC_IND: "'Patient ID', 'Specimen', 'Sample N', 'Population', 'Pregnancy stage', 'Pediatric/Gestational age', 'Sample time', 'Time unit', 'Note'",
    PROMPTS_NAME_PK_DRUG_IND: "'Patient ID', 'Drug/Metabolite name', 'Dose amount', 'Dose unit', 'Dose frequency', 'Dose schedule', 'Dose route', 'Population', 'Pregnancy stage', 'Pediatric/Gestational age', 'Note'",
    PROMPTS_NAME_PK_POPU_IND: "'Patient ID', 'Characteristic', 'Characteristic subcategory', 'Characteristic unit', 'Characteristic value', 'Population', 'Pregnancy stage', 'Pediatric/Gestational age', 'Note'",
    PROMPTS_NAME_PE_STUDY_INFO: "'Study type', 'Population', 'Study design', 'Pregnancy stage', 'Drug name', 'Data source', 'Inclusion criteria', 'Exclusion criteria', 'Outcomes', 'Subject N'",
    PROMPTS_NAME_PE_STUDY_OUT: "'Characteristic', 'Exposure', 'Outcome', 'Parameter unit', 'Parameter statistic', 'Parameter value', 'Variation type', 'Variation value', 'Interval type', 'Lower bound', 'Upper bound', 'P value'"
}


def prepare_starter_history(
    task_type: str,
    input_article_info: str,
    transaction_name: str,
    curated_result: str,
    reasoning_trace: str,
    system_intro: str = INTRODUCTION,
    system_rules: str = REGULATION
) -> list[dict]:
    """
    Prepares the initial conversation history for a follow-up curation session.
    """

    # Compose task-specific expected column headers
    task_instruction = f"\n\nYour assigned task is to extract tabular information from the paper with the following columns:\n{COLUMN_DEFINITIONS.get(task_type, '')}"

    # Compose full prompt context
    input_description = "\n\nYour user has provided the following input article:\n" + input_article_info
    reasoning_summary = f"""\n\nBased on your understanding of the input, you followed the reasoning process below:\n{reasoning_trace}\n\n
As a result, you produced the following curated output:\n\n{curated_result}\n\n"""

    # Starting message for follow-up
    assistant_greeting = (
        "Hello! You are following up on the curation task: <br>"
        "<strong>" + transaction_name + "</strong>. <br>"
        "Feel free to ask about the result or reasoning."
    )
    starter_history = [
        {
            "role": "system",
            "content": system_intro + input_description + task_instruction + reasoning_summary + "\n\n" + system_rules
        },
        {
            "role": "assistant",
            "content": assistant_greeting
        }
    ]

    return starter_history
