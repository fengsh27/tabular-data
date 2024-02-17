from typing import List, Any
from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def request_paper(pmid: str):
    pmid = pmid.strip()
    ua = UserAgent()
    print(ua.chrome)
    header = {'User-Agent':str(ua.chrome)}
    print(header)
    
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/"
    r = requests.get(url, headers=header)
    if r.status_code != 200:
        print(f"{r.status_code}: {r.reason}")
        logger.error(f"{r.status_code}: {r.reason}")
        return (False, r.reason, r.status_code)
    
    return (True, r.text, r.status_code)

def extract_tables_from_html(html_content: str) -> List[Any]:
    soup = BeautifulSoup(html_content, "html.parser")
    tags = soup.select("div.table-wrap.anchored.whole_rhythm")
    tables = []
    for tag in tags:
        tbl_soup = BeautifulSoup(str(tag), "html.parser")
        caption = tbl_soup.select("div.caption")
        caption = caption[0].text if len(caption) > 0 else ""
        table = tbl_soup.select("div.xtable")
        table = str(table[0]) if len(table) > 0 else ""
        table = convert_table_to_dataframe(table)
        footnote = tbl_soup.select("div.tblwrap-foot")
        footnote = footnote[0].text if len(footnote) > 0 else ""
        tables.append({"caption": caption, "table": table, "footnote": footnote})

    return tables

def convert_table_to_dataframe(table: str):
    try:
        df = pd.read_html(table)
        return df[0]
    except Exception as e:
        logger.error(e)
        print(e)
        return None


if __name__ == "__main__":
    (res, content, status_code) = request_paper("23106931")
    if res:
        tables = extract_tables_from_html(content)
        for tbl in tables:
            print(tbl["caption"])
            df = convert_table_to_dataframe(tbl["table"])
            print(df)
            print(tbl["footnote"])


