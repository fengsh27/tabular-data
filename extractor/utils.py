import json
from typing import Optional
from bs4 import BeautifulSoup
import functools
import html
import urllib
import logging
import pandas as pd
import csv
from io import StringIO
import re
import math

logger = logging.getLogger(__name__)


def decode_url(url_str: str) -> str:
    str1 = html.unescape(url_str)
    str2 = urllib.parse.unquote_plus(str1)
    while str1 != str2:
        str1 = str2
        str2 = urllib.parse.unquote_plus(str1)
    return str2


def convert_html_table_to_dataframe(table: str):
    try:
        # remove unicode \xa0 (browser space) that can be recognized by streamlit.dataframe()
        table = table.replace("\xa0", " ")
        # Remove non-standard rowspan and colspan attributes (e.g., rowspan="50%")
        table = re.sub(r'\s(rowspan|colspan)="\d+%"', '', table)
        table_io = StringIO(table)
        df = pd.read_html(table_io, keep_default_na=False)
        return df[0]
    except Exception as e:
        logger.error(e)
        print(e)
        return None


def preprocess_csv_table_string(table: str):
    """
    This function is to pre-process csv table string to make it able to be read
    as DataFrame.
    1). Check if there are redundant comma at the end of each row, if yes, remove
    the comma
    """
    try:
        csv_data = StringIO(table)
        csv_reader = csv.reader(csv_data)
        ix = 0
        item_count = 0
        row_strs = []
        # check if it need to process csv table string (remove redundant empty column at the end of row)
        for row in csv_reader:
            ix += 1
            if ix == 1:  # header
                item_count = len(row)
                row_strs.append(",".join(row))
                continue
            row_count = len(row)
            if row_count > item_count and len(row[row_count - 1].strip()) == 0:
                row = row[:-1]  # remove empty column at the end of row
            row_strs.append(",".join(row))
        return ("\n".join(row_strs)) + "\n"
    except csv.Error as e:
        logger.error(str(e))
        return table


def convert_csv_table_to_dataframe(table: str):
    try:
        # remove redudant comma at the end of row
        modified_str = preprocess_csv_table_string(table)
        csv_data = StringIO(modified_str)
        df = pd.read_csv(csv_data, sep=",")
        return df
    except Exception as e:
        logger.error(e)
        return None


def convert_html_to_text(html_content: str) -> str:
    """
    This function is used to convert html string to text, that is,
    extract text from html content, including tables.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text


def convert_html_to_text_no_table(html_content: str) -> str:
    """
    Yichuan: convert_html_to_text, but no table
    """
    soup = BeautifulSoup(html_content, "html.parser")
    for table in soup.find_all("table"):
        table.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text


def remove_references(text: str):
    ix = text.lower().rfind("references")
    if ix < 0:
        logger.warn("Can't find 'References' in paper")
        return text
    return text[:ix]


def escape_markdown(content: str) -> str:
    content = content.replace("#", "\\#")
    content = content.replace("*", "\\*")
    return content


def is_valid_csv_table(tbl_str):
    try:
        csv_data = StringIO(tbl_str)
        csv_reader = csv.reader(csv_data)
        for row in csv_reader:
            pass
        return True
    except csv.Error:
        return False


TITLE_MAX_LENGTH = 50


def extract_table_title(table: dict):
    if "caption" not in table or len(table["caption"]) == 0:
        return None
    cap: str = table["caption"]
    cap = cap.strip()
    return cap if len(cap) < TITLE_MAX_LENGTH else cap[: TITLE_MAX_LENGTH - 2] + " ..."


NUMBER_RE_PATTERN = r"^[-+]?\d{0,3}(,\d{3})*(\.\d+)?$"


def remove_comma_in_number_string(content: str) -> str:
    """
    This function is to remove comma in number, like 1,234 -> 1234
    """
    content = content.strip()
    # regular expression for number
    if content.startswith(","):
        return "," + remove_comma_in_number_string(content[1:])
    pattern = NUMBER_RE_PATTERN
    if re.match(pattern, content):
        return content.replace(",", "")
    else:
        return content


def remove_comma_in_string(content: str, repl: Optional[str] = " ") -> str:
    """
    This function is to remove comman in string, which will make it failed to
    read csv string
    """
    if "," in content:
        return content.replace(",", repl)
    return content


def _find_comma_and_right_brace(content: str, ix: int):
    length = len(content)
    found = False
    while ix >= 0:
        if content[ix] == ",":
            jx = ix - 1
            while jx >= 0:
                if content[jx] == "}":
                    found = True
                    break
                elif content[jx] != " ":
                    break
                jx -= 1
            return found
        elif content[ix] == " ":
            ix -= 1
            continue
        else:
            break

    return found


def _truncate_json_content(content: str):
    """
    This function is to remove imcompleted json object, like this:
    '[{"a": "...", "b": "...", "c": "..."}, {"a": "...",' will be truncated to
    '[{"a": "...", "b": "...", "c": "..."},'
    """
    length = len(content)
    found = False
    for rev_ix in range(length):
        ix = length - rev_ix - 1
        if content[ix] == "{":
            found = _find_comma_and_right_brace(content, ix - 1)
            if found:
                break

    return (content[:ix], found) if found else (content, found)


def _strip_contents(contents: list[str]):
    res = []
    for ix in range(len(contents)):
        if contents[ix] is None:
            continue
        cont = contents[ix].strip()
        if cont.startswith("```json"):
            cont = cont[7:]
            cont = cont.strip()
        if cont.endswith("```"):
            cont = cont[:-3]
        res.append(cont)
    return res


def concate_llm_contents(contents: list[str], usages: list[int]):
    contents = _strip_contents(contents)

    truncated = False
    for ix in range(len(contents)):
        if ix == 0:
            continue
        if contents[ix].startswith("[{"):
            contents[ix - 1], truncated = _truncate_json_content(contents[ix - 1])
            contents[ix] = contents[ix][1:]  # remove '['

    all_usages = functools.reduce(lambda a, b: a + b, usages)

    all_contents = "".join(contents)
    MAX_RETRIES = 5
    retries = 0
    while retries < MAX_RETRIES:
        try:
            json.loads(all_contents)
            break
        except json.JSONDecodeError as e:
            retries += 1
            col = e.colno
            length = 0
            if len(contents) == 1:
                all_contents, truncated = _truncate_json_content(contents[0])
                all_contents = all_contents.strip()
                if all_contents[-1] == ",":
                    all_contents = all_contents[:-1]
                if all_contents[-1] != "]":
                    all_contents = all_contents + "]"
            else:
                for ix in range(len(contents) - 1):
                    cont = contents[ix]
                    length += len(cont)
                    next_length = len(contents[ix + 1])
                    if col >= length and col < length + next_length:
                        # process the {ix}th content
                        contents[ix], truncated = _truncate_json_content(contents[ix])

                        # process the {ix+1}the content,
                        left_brace_ix = contents[ix + 1].find("{")
                        contents[ix + 1] = contents[ix + 1][left_brace_ix:]
                        if (
                            left_brace_ix == -1
                            and contents[ix] is not None
                            and contents[ix][-1] == ","
                        ):
                            contents[ix] = contents[ix][:-1]
                        truncated = True
                        break
                all_contents = "".join(contents)
        except Exception:
            break

    return all_contents, all_usages, truncated


def extract_float_value(s) -> float:
    if not isinstance(s, str):
        try: 
            return float(s)
        except (
            ValueError, 
            TypeError
        ):
            return math.nan
    pattern = r"([-+]?[0-9]*\.?[0-9]+)"
    match = re.search(pattern, s)
    if match:
        return float(match.group())
    else:
        return None


def extract_float_values(s) -> list[float]:
    if not isinstance(s, str):
        try: 
            return [float(s),]
        except (
            ValueError, 
            TypeError
        ):
            return [math.nan,]
    pattern = r"([-+]?[0-9]*\.?[0-9]+)"
    match = re.findall(pattern, s)
    if match:
        ret_arr = []
        for val in match:
            ret_arr.append(float(val))
        return ret_arr
    else:
        return None

def escape_braces_for_format(text: str) -> str:
    # replace all {xxx} by {{xxx}}
    # return re.sub(r'\{([^{}]+)\}', r'{{\1}}', text)
    
    # First replace single } not part of }} with }}
    text = re.sub(r'(?<!})}(?!})', '}}', text)
    # Then replace single { not part of {{
    text = re.sub(r'(?<!{){(?!{)', '{{', text)
    return text

def convert_sections_to_full_text(sections: list[dict[str, str]]) -> str:
    article_text = "\n".join(
        f"{sec['section']}\n{sec['content']}" for sec in sections
    )
    
    article_text = convert_html_to_text_no_table(article_text)
    article_text = remove_references(article_text)
    return article_text

