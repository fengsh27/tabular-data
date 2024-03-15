
from typing import Dict, Any, List
from pandas import DataFrame

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
