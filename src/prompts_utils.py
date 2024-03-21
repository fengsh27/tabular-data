
from typing import Dict, Any, List
from pandas import DataFrame

DEFAULT_PROMPTS = """
Please act as a biomedical assistant, extract the following information from the provided biomedical paper and output as a table in markdown format:
1. Drug name, the name of drug mentioned in the paper
2. Specimen, what is the specimen, like "blood", "breast milk", "cord blood", and so on.
3. Pregnancy Stage, pregnancy stage, What pregnancy stages of patients mentioned in the paper, like "postpartum", "before pregnancy", "1st trimester" and so on. If not mentioned, please label as "N/A",
4. Parameter type, the type of parameter, like "concentration after the first dose", "concentration after the second dose", "clearance", "Total area under curve" and so on.
5. Value, the value of parameter
6. unit,  the unit of the value
7. Summary Statistics, the statistics method to summary the data, like "geometric mean", "arithmetic mean"
8. Interval type, specifies the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like "95% CI", "range" and so on.
9. lower limit, the lower bounds of the interval
10.  high limit. the higher bounds of the interval
11. Population: Describe the patient age distribution, including categories such as "pediatric," "adults," "old adults," "maternal," "fetal," "neonate," etc.


Please note: 

1. Only output markdown table without any other characters and embed the text in code chunks, so it won't convert to HTML in the assistant.
2. Ensure to extract all available information for each field without omitting any details.
3. If the information that is not provided, please leave it empty 
"""

def _generate_table_prompts(tbl: Dict[str, str | DataFrame]):
    raw_tag = tbl.get("raw_tag", None)
    if raw_tag is not None:
        table_text = f"html table is:\n{raw_tag}"
        return table_text
    caption = tbl.get("caption", None)
    table = tbl.get("table", None)
    footnote = tbl.get("footnote", None)
    table_text = ""
    if caption is not None:
        table_text += f"table caption: {caption}\n"
    if table is not None:
        table_text += "table is:\n"
        table_text += table.to_csv()
        table_text += "\n"
    if footnote is not None:
        table_text += f"table footnote: {footnote}\n"
    return table_text

def generate_tables_prompts(tables: List[Dict[str, str|DataFrame]]):
    prompts = "Here are the tables in the paper (including their caption and footnote)\n"
    for table in tables:
        prompts += _generate_table_prompts(table)
        prompts += "\n"
    return prompts

def generate_paper_text_prompts(text: str):
    return f"Here is the paper:\n {text}"

def generate_question(source: str):
    return f"Now please extract information from {source} and output to a table in markdown format"
