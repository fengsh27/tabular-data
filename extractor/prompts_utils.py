
from typing import Dict, Any, List, Optional
from pandas import DataFrame
import json
import logging

from extractor.constants import PROMPTS_NAME_PK

logger = logging.getLogger(__name__)

PKSUMMARY_TABLE_ROLE_PROMPTS = "Please act as a biomedical assistant, help to extract the information from the provided source"
PKSUMMARY_TABLE_SOURCE_PROMPTS = "html table from biomedical article"
PKSUMMARY_TABLE_OUTPUT_COLUMNS = [
    "Drug name",
    "Analyte",
    "Specimen", 
    "Population", 
    "Pregnancy stage", 
    "Subject N",
    "Parameter type", 
    "Value", 
    "Unit", 
    "Summary statistics", 
    "Variation type",
    "Variation value",
    "Interval type", 
    "Lower limit",
    "High limit", 
    "P value",
]
PKSUMMARY_TABLE_OUTPUT_COLUMNS_DEFINITION = [
    "Drug name: is text, the name of drug mentioned in the paper",
    "Analyte: is text, either the primary drug, its metabolite, or another drug on which the primary drug acts.",
    "Specimen: is text, what is the specimen, like 'blood', 'breast milk', 'cord blood', and so on.",
    "Pregnancy stage: is text, What pregnancy stages of patients mentioned in the paper, like 'postpartum', 'before pregnancy', '1st trimester' and so on. If not mentioned, please label as 'N/A',",
    "Parameter type: is text, the type of parameter, like 'concentration after the first dose', 'concentration after the second dose', 'clearance', 'Total area under curve' and so on.",
    "Value: is a number, the value of parameter",
    "Unit: the unit of the value",
    "Summary statistics: the statistics method to summary the data, like 'geometric mean', 'arithmetic mean'",
    "Interval type: is text, specifies the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like '95% CI', 'range' and so on.",
    "Lower limit: is a number, the lower bounds of the interval",
    "Population: Describe the patient age distribution, including categories such as 'pediatric,' 'adults,' 'old adults,' 'maternal,' 'fetal,' 'neonate,' etc.",
    "High limit: is a number, the higher bounds of the interval",
    "Subject N:  the number of subjects that correspond to the specific parameter. ",
    "Variation value: is a number, the number that corresponds to the specific variation.", 
    "Variation type: the variability measure (describes how spread out the data is) associated with the specific parameter, e.g., standard deviation (SD), CV%.",
    "P value: The p-value is a number, calculated from a statistical test, that describes the likelihood of a particular set of observations if the null hypothesis were true; varies depending on the study, and therefore it may not always be reported."
]
PKSUMMARY_TABLE_OUTPUT_NOTES = [
    "1. Only output table in json format without any other characters, no triple backticks ``` and no 'json'.",
    "2. Ensure to extract all available information for each field without omitting any details.",
    "3. If the information that is not provided, please leave it with empty string."
]

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
    return f"Now please extract information from {source} and output to a table string in json format"


class GeneratedTableProcessor(object):
    """
    This class is to process the content generated by LLM:
    1. check the format of the content.
    2. If the content is not in csv format, we will convert to csv format
    """
    def __init__(
        self, 
        delimiter: Optional[str] = ', ', 
        columns: Optional[List[str]] = PKSUMMARY_TABLE_OUTPUT_COLUMNS
    ):
        self.delimiter = delimiter
        self.columns = columns
        self.lower_columns = list(map(lambda x: x.lower(), self.columns))

    def process_content(self, content: str) -> str:
        if self._check_content_format(content) == "json":
            return self._convert_json_to_csv(content)
        else:
            return content
        
    def _convert_to_csv_header(self, headers) -> str:
        csv_headers = self.delimiter.join(headers)
        return csv_headers
    
    def _convert_to_csv_row(self, row: Dict[str, Any]) -> str:
        lower_dict = {}
        for k in row:
            lower_dict[k.lower()] = row[k]
        vals = ""
        col_cnt = len(self.lower_columns)
        for ix, col in enumerate(self.lower_columns):
            if col in lower_dict:
                vals += f"{lower_dict[col]}"
            if ix < col_cnt-1:
                vals += self.delimiter
        return vals
    
    def _convert_json_to_csv(self, content: str) -> str | None:
        stripped_content = content.strip()
        if stripped_content.startswith('['):
            stripped_content = '{' + f'"content": {stripped_content}' + '}'
        try:
            json_obj = json.loads(stripped_content)
            csv_str = self._convert_to_csv_header(PKSUMMARY_TABLE_OUTPUT_COLUMNS)
            csv_str += "\n"
            rows: List = json_obj["content"]
            for ix, row in enumerate(rows):
                csv_str += self._convert_to_csv_row(row)
                csv_str += "\n"
            return csv_str
        except Exception as e:
            logger.error(e)
            return None

    def _check_content_format(self, content: str) -> str:
        stripped_content = content.strip()
        if stripped_content.startswith('[') or stripped_content.startswith('{'):
            return "json"
        else:
            return "csv"
