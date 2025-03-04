

import json
import re
from typing import Any, Dict, List, Optional
import logging

from extractor.utils import (
    remove_comma_in_number_string, 
    remove_comma_in_string,
)

logger = logging.getLogger(__name__)

from extractor.constants import (
    PROMPTS_NAME_PK,
    PROMPTS_NAME_PE,
)

class JsonEnclosePropertyNameInQuotesPlugin:
    """
    This plugin is to ensure in json, the property names are enclosed in quotes ""
    """
    def __init__(self, prompts_type: str=PROMPTS_NAME_PK) -> None:
        self.prompts_type = prompts_type
        with open("./prompts/pk_prompts.json", "r") as pk_obj:
            pk_str = pk_obj.read()
            self.pk_prompts = json.loads(pk_str)
        with open("./prompts/pe_prompts.json", "r") as pe_obj:
            pe_str = pe_obj.read()
            self.pe_prompts = json.loads(pe_str)
    
    def process(self, json_str: str):
        prompts = self.pk_prompts if self.prompts_type == PROMPTS_NAME_PK else self.pe_prompts
        column_names = prompts["table_extraction_prompts"]["output_columns"]
        for cn in column_names:
            json_str = re.sub(rf"(?<![a-zA-Z0-9]){cn}:", f'"{cn}":', json_str)
        
        return json_str

class JsonFieldValidatePlugin:
    """
    This plugin is to eliminate invalid fields like the following:
    {"DN":"lorazepam","Ana":"lorazepam","Sp":"cord blood","Pop":"fetal","PS":"N/A","SN":"1","PT":"concentration",
    "V":5.77,"U":"ng/ml","","","","","","","",""}
    """
    def __init__(self):
        pass

    @staticmethod
    def _clean_json_string(json_str: str) -> str:
        cleaned_json_str = re.sub(r',\s*""\s*', '', json_str)
        cleaned_json_str = re.sub(r'{\s*""\s*,', '{', cleaned_json_str)
        cleaned_json_str = re.sub(r',\s*""\s*}', '}', cleaned_json_str)
        
        return cleaned_json_str

    def process(self, json_str: str):
        return JsonFieldValidatePlugin._clean_json_string(json_str)
        
class GeneratedPKSummaryTableProcessor(object):
    """
    This class is to process the content generated by LLM:
    1. check the format of the content.
    2. If the content is not in csv format, we will convert to csv format
    """
    def __init__(
        self, 
        prompts_type: Optional[str]=PROMPTS_NAME_PK,
        delimiter: Optional[str] = ', ', 
    ):
        self.prompt_type = prompts_type
        self.delimiter = delimiter
        temp_columns, temp_columns_dict = self._get_prompts_columns_and_columns_dict()
        self.columns = temp_columns
        self.lower_columns = list(map(lambda x: x.lower(), self.columns))
        self.columns_dict = temp_columns_dict
        self.plugins = [
            JsonEnclosePropertyNameInQuotesPlugin(prompts_type=prompts_type),
            JsonFieldValidatePlugin(),
        ]

    def process_content(self, content: str, has_header=True) -> str:
        content = self._strip_table_content(content)
        
        for preprocessor in self.plugins:
            content = preprocessor.process(content)

        if self._check_content_format(content) == "json":
            return self._convert_json_to_csv(content, has_header)
        else:
            return content
        
    def _get_prompts_columns_and_columns_dict(self):
          fn = ("./prompts/pk_prompts.json"
                if self.prompt_type == PROMPTS_NAME_PK 
                else "./prompts/pe_prompts.json")
          with open(fn, "r") as fobj:
              json_obj = json.load(fobj)
              return (
                  json_obj["table_extraction_prompts"]["output_columns"], 
                  json_obj["table_extraction_prompts"]["output_columns_map"]
              )
    def _get_fullname_headers(self)->List[str]:
        return list(map(lambda item: item[1], self.columns_dict))
    
    def _convert_to_csv_header(self) -> str:
        headers = self._get_fullname_headers()
        csv_headers = self.delimiter.join(headers)
        return csv_headers  
    
    def _convert_to_csv_row(self, row: Dict[str, Any]) -> str:
        lower_key_row = {}
        for k in row:
            lower_key_row[k.lower()] = row[k]
        vals = ""
        col_cnt = len(self.lower_columns)
        for ix, col in enumerate(self.lower_columns):
            if col in lower_key_row:
                the_val = lower_key_row[col]
                the_val = f"{the_val}"                
                the_val = remove_comma_in_number_string(the_val)
                the_val = remove_comma_in_string(the_val)
                vals += the_val
            vals += self.delimiter
        vals = vals[:-2] # remove the last delimiter
        return vals
    
    def _convert_json_to_csv(self, content: str, has_header=True) -> str | None:
        stripped_content = content.strip()
        if stripped_content.startswith('['):
            json_content = '{' + f'"content": {stripped_content}' + '}'
        try:
            json_obj = json.loads(json_content)
            csv_str = self._convert_to_csv_header() + "\n" if has_header else ''
            rows: List = json_obj["content"]
            for ix, row in enumerate(rows):
                val = self._convert_to_csv_row(row)
                csv_str += val
                csv_str += "\n"
            return csv_str
        except Exception as e:
            logger.error(e)
            raise e
    """
    def _try_to_convert_incompleted_json_to_csv(self, content):
        processing = []
        processed = []
        for ix in range(len(content)):
            char = content[ix]
            if char == "{":
                prev = processing[-1]
                level = prev[1] + 1 if prev[0] == '{' else 0
                processing.append([char, level, ix])
    """
    def _strip_table_content(self, content: str) -> str:
        """
        This function is to remove redundant characters, like white spaces, 
        ```json ... ``` or ```csv ... ```
        """
        strp_content = content.strip()
        ix = strp_content.find("```json")
        if ix >= 0:
            strp_content = strp_content[ix+7:]
        ix = strp_content.find("```csv")
        if ix >= 0:
            strp_content = strp_content[ix+6:]
        if strp_content.startswith("```"):
            strp_content = strp_content[3:]
        if strp_content.endswith("```"):
            strp_content = strp_content[:-3]
        return strp_content.strip()
    
    def _check_content_format(self, content: str) -> str:
        stripped_content = content.strip()
        if stripped_content.startswith('[') or stripped_content.startswith('{'):
            return "json"
        else:
            return "csv"
