
from typing import Optional, Tuple
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import html
import urllib
import logging
import pandas as pd
import csv
from io import StringIO
import re

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
        df = pd.read_html(table)
        return df[0]
    except Exception as e:
        logger.error(e)
        print(e)
        return None
    
def convert_csv_table_to_dataframe(table: str):
    try:
        # first, let me handle the numbers which have commas in them
        pattern = r'\b\d{1,3}(,\d{3})\b'
        modified_str = re.sub(pattern, lambda match: f'"{match.group(0)}"', table)
        csv_data = StringIO(modified_str)
        df = pd.read_csv(csv_data, sep=',')
        return df
    except Exception as e:
        logger.error(e)
        return None

def convert_html_to_text(html_content: str) -> str:
    '''
    This function is used to convert html string to text, that is,
    extract text from html content, including tables.
    '''
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text

def remove_references(text: str):
    ix = text.lower().rfind("references")
    if ix < 0:
        logger.warn(f"Can't find 'References' in paper")
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