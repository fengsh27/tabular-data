from typing import List, Any
from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
import logging
import urllib.parse

logger = logging.getLogger(__name__)

FULL_TEXT_LENGTH_THRESHOLD = 10000 # we assume the length of full-text paper should be 
                                   # greater than 10000
MAX_FULL_TEXT_LENGTH = 31 * 1024   # should not be greater than 31K

def _request_by_abstract_page(pmid: str):
    ua = UserAgent()
    header = {'User-Agent': str(ua.chrome)}
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    r = requests.get(url, headers=header)
    if r.status_code != 200:
        return (False, "", r.status_code)
    html_content = r.text

    # extract full-text link
    soup = BeautifulSoup(html_content, "html.parser")
    tags = soup.select("div.full-view > div.full-text-links-list > a.link-item")
    if tags is None or len(tags) == 0:
        return (False, "Can't get full-text url by selector", -1)
    aTag = tags[0]
    full_text_url = aTag.attrs.get("href", None)
    if full_text_url is None:
        return (False, "Can't get full-text url from href attribute", -1)
    
    # request from full-text link
    r = requests.get(full_text_url, headers=header)
    if r.status_code != 200:
        return (False, "Can't access full-text-rul", r.status_code)
    
    # check if there is redirect link in it
    content_length = len(r.text) # check the length of full-text
    soup = BeautifulSoup(r.text, "html.parser")
    tag: Tag = soup.find(id="redirectURL")
    if ( content_length < FULL_TEXT_LENGTH_THRESHOLD and
        tag is not None):
        redirect_url = tag.get("value", None)
        if redirect_url is None:
            return (False, "Can't accesss redirect url", -1)
        r = requests.get(redirect_url, headers=header)
        if r.status_code != 200:
            return (False, "Failed to access redirect url", r.status_code)
        return (True, r.text, r.status_code)

    return (True, r.text, r.status_code)

def request_paper(pmid: str):
    pmid = pmid.strip()
    ua = UserAgent()
    print(ua.chrome)
    header = {'User-Agent':str(ua.chrome)}
    print(header)
    
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/"
    r = requests.get(url, headers=header)
    if r.status_code == 200:
        return (True, r.text, r.status_code)
    
    # Maybe no full-text in PubMed, in this case, we will try
    # full-text link in abstract page (https://pubmed.ncbi.nlm.nih.gov/{pmid}/)
    print(f"{r.status_code}: {r.reason}")
    logger.warn(f"{r.status_code}: {r.reason}")
    (res, text, code) = _request_by_abstract_page(pmid)

    return (res, text, code)

def convert_html_to_text(html_content: str) -> str:
    '''
    This function is used to convert html string to text, that is,
    extract text from html content, including tables.
    '''
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n", strip=True)

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

def test_request_paper_and_obtain_tables():
    pmid = "23106931"
    (res, content, status_code) = request_paper(pmid)
    if res:
        tables = extract_tables_from_html(content)
        for tbl in tables:
            print(tbl["caption"])
            df = convert_table_to_dataframe(tbl["table"])
            print(df)
            print(tbl["footnote"])

def test_request_paper_by_full_text_link():
    # pmid = "30322724"
    # pmid = "24785239"
    pmid = "30284023"
    (res, content, status_code) = request_paper(pmid)
    text = convert_html_to_text(content)
    ix = text.find("References")
    content = text[:ix]
    assert res



if __name__ == "__main__":
    test_request_paper_by_full_text_link()


