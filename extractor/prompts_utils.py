
from typing import Dict, Any, List, Optional
from pandas import DataFrame
import json
import logging

from extractor.constants import (
    PKSUMMARY_TABLE_OUTPUT_COLUMNS, 
    PKSUMMARY_TABLE_OUTPUT_COLUMNS_DEFINITION, 
    PKSUMMARY_TABLE_OUTPUT_NOTES, 
    PKSUMMARY_TABLE_ROLE_PROMPTS, 
    PKSUMMARY_TABLE_SOURCE_PROMPTS, 
    PROMPTS_NAME_PK,
)

logger = logging.getLogger(__name__)

class TableExtractionPKSummaryPromptsGenerator(object):
    def __init__(self):
        pass

    def _read_prompts_config_file(self, prompts_name):
        if prompts_name == PROMPTS_NAME_PK:
            return open("./prompts/pk_prompts.json", "r")
        else:
            return open("./prompts/pe_prompts.json", "r")

    @staticmethod
    def _prompts_to_str(prmpt: str | list[str], delimiter: Optional[str]="\n"):
        return (
            delimiter.join(prmpt) 
            if isinstance(prmpt, list)
            else str(prmpt)
        )
    @staticmethod
    def _generate_prompts(
        role: str, 
        source: str, 
        output_columns: str | list[str],
        output_columns_def: str | list[str],
        output_notes: str | list[str]
    ):
        if isinstance(output_columns, list) and \
            isinstance(output_columns_def, list) and \
            len(output_columns) == len(output_columns_def):
            name = output_columns[0].strip()
            col_def = output_columns_def[0].strip()
            col_def_name = col_def[:len(name)]
            if col_def_name.lower() != name.lower():
                # integrate column names to column definitions
                for ix in range(len(output_columns)):
                    name = output_columns[ix]
                    output_columns_def[ix] = f"{name}: {output_columns_def[ix]}"

        output_columns = \
            TableExtractionPKSummaryPromptsGenerator._prompts_to_str(output_columns, delimiter=',')
        output_columns_def = \
            TableExtractionPKSummaryPromptsGenerator._prompts_to_str(output_columns_def)
        output_notes = \
            TableExtractionPKSummaryPromptsGenerator._prompts_to_str(output_notes)
        return "\n".join([
            f"{role}\nThe source is {source}.",
            f"Here is desired output columns:{output_columns}",
            f"Here is output column description: \n{output_columns_def}",
            f"Please Notes: \n{output_notes}"
        ])


    @staticmethod
    def _generate_system_prompts_by_default():
        return TableExtractionPKSummaryPromptsGenerator._generate_prompts(
            PKSUMMARY_TABLE_ROLE_PROMPTS,
            PKSUMMARY_TABLE_SOURCE_PROMPTS,
            PKSUMMARY_TABLE_OUTPUT_COLUMNS,
            PKSUMMARY_TABLE_OUTPUT_COLUMNS_DEFINITION,
            PKSUMMARY_TABLE_OUTPUT_NOTES
        )
    def get_prompts_file_content(
        self, 
        prompts_name: str,
        json_beautifying: Optional[bool] = False
    ):
        try:
            fobj = self._read_prompts_config_file(prompts_name)
            if not json_beautifying:
                return fobj.read()
            else:
                json_obj = json.load(fobj)
                return json.dumps(json_obj, indent=4)
        except Exception as e:
            logger.error(e)
            return f"Unknown error occurred: {e}"
        finally:
            fobj.close()
    def generate_system_prompts(self, prompts_name: str):
        try: 
            fobj = self._read_prompts_config_file(prompts_name)
            content = json.load(fobj)
            table_prompts: Optional[Dict] = content.get("table_extraction_prompts", None)
            if table_prompts is None:
                return TableExtractionPKSummaryPromptsGenerator._generate_system_prompts_by_default()
            role = table_prompts.get("role_description", PKSUMMARY_TABLE_ROLE_PROMPTS)
            source = table_prompts.get("source", PKSUMMARY_TABLE_SOURCE_PROMPTS)
            output_columns = table_prompts.get("output_columns", PKSUMMARY_TABLE_OUTPUT_COLUMNS)
            output_columns_def = table_prompts.get(
                "output_column_definitions", 
                PKSUMMARY_TABLE_OUTPUT_COLUMNS_DEFINITION
            )
            output_notes = table_prompts.get("output_notes", PKSUMMARY_TABLE_OUTPUT_NOTES)
            return TableExtractionPKSummaryPromptsGenerator._generate_prompts(
                role, source, output_columns, output_columns_def, output_notes
            )
        except Exception as e:
            logger.error(e)
            return self._generate_system_prompts_by_default()
        finally:
            fobj.close()

def _generate_table_prompts(tbl: Dict[str, str | DataFrame]):
    raw_tag = tbl.get("raw_tag", None)
    if raw_tag is not None:
        table_text = f"html table is:\n```\n{raw_tag}```\n"
        return table_text
    caption = tbl.get("caption", None)
    table = tbl.get("table", None)
    footnote = tbl.get("footnote", None)
    table_text = ""
    if caption is not None:
        table_text += f"table caption: {caption}\n"
    if table is not None:
        table_text += "table is:\n```\n"
        table_text += table.to_csv()
        table_text += "\n```\n"
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
    return f"Now please extract information from {source} and output to a table string in compact json format"

