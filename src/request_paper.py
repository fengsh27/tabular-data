from typing import List, Any, Tuple
from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
import logging
import urllib.parse
import html

logger = logging.getLogger(__name__)

FULL_TEXT_LENGTH_THRESHOLD = 10000 # we assume the length of full-text paper should be 
                                   # greater than 10000
MAX_FULL_TEXT_LENGTH = 31 * 1024   # should not be greater than 31K
cookies = {
    'cookie_name': 'cookie_value'
}
headers = {
    'authority': 'www.google.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    'cookie': "SID=ZAjX93QUU1NMI2Ztt_dmL9YRSRW84IvHQwRrSe1lYhIZncwY4QYs0J60X1WvNumDBjmqCA.; __Secure-", 
    #..,
    'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
    'sec-ch-ua-arch': '"x86"',
    'sec-ch-ua-bitness': '"64"',
    'sec-ch-ua-full-version': '"115.0.5790.110"',
    'sec-ch-ua-full-version-list': '"Not/A)Brand";v="99.0.0.0", "Google Chrome";v="115.0.5790.110", "Chromium";v="115.0.5790.110"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-model': '""',
    'sec-ch-ua-platform': 'Windows',
    'sec-ch-ua-platform-version': '15.0.0',
    'sec-ch-ua-wow64': '?0',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'x-client-data': '#..',
}

def _decode_url(url_str: str) -> str:
    str1 = html.unescape(url_str)
    str2 = urllib.parse.unquote_plus(str1)
    while str1 != str2:
        str1 = str2
        str2 = urllib.parse.unquote_plus(str1)
    return str2

def _request_by_curl(url:str) -> Tuple[bool, str, int]:
    from subprocess import Popen, PIPE
    args = ["curl", url]
    prc = Popen(args, shell=False, stdout=PIPE, stderr=PIPE)
    stdout, stderr = prc.communicate()
    content = str(stdout)
    soup = BeautifulSoup(content, "html.parser")
    aTag = soup.find("a")
    if aTag is None:
        return (False, "Can't obtain redirect url", -1)
    redirect_url = aTag.get("href")
    return _request_from_full_text_url(redirect_url)

def _request_from_full_text_url(full_text_url: str) -> Tuple[bool, str, int]:
    ua = UserAgent()
    header = headers
    header["User-Agent"] = str(ua.chrome)
    r = requests.get(full_text_url, headers=header, allow_redirects=True, cookies=cookies)
    if r.status_code != 200:
        # failed to get html content, maybe it only allow redirect access
        if r.status_code != 403: # Unknown reason
            return (False, "Can't access full-text url", r.status_code)
        # try to handle redirect url
        return _request_by_curl(full_text_url)
    
    # successfully get html context, check if there is redirect link in it
    content_length = len(r.text) # check the length of full-text
    soup = BeautifulSoup(r.text, "html.parser")
    tag: Tag = soup.find(id="redirectURL")
    if ( content_length < FULL_TEXT_LENGTH_THRESHOLD and
        tag is not None):
        redirect_url = tag.get("value", None)
        if redirect_url is None:
            return (False, "Can't accesss redirect url", -1)
        redirect_url = _decode_url(redirect_url)
        r = requests.get(redirect_url, headers=header, allow_redirects=True, cookies=cookies)
        if r.status_code != 200:
            return (False, "Failed to access redirect url", r.status_code)
        return (True, r.text, r.status_code)
    else:
        return (True, r.text, r.status_code)

def _extract_full_text_link(html_content: str) -> Tuple[bool, str, int]:
    soup = BeautifulSoup(html_content, "html.parser")
    tags = soup.select("div.full-view > div.full-text-links-list > a.link-item")
    if tags is None or len(tags) == 0:
        return (False, "Can't get full-text url by selector", -1)
    aTag = tags[0]
    full_text_url = aTag.attrs.get("href", None)
    if full_text_url is None:
        return (False, "Can't get full-text url from href attribute", -1)
    return (True, full_text_url, 200)

def _request_by_abstract_page(pmid: str):
    ua = UserAgent()
    header = headers
    header["User-Agent"] = str(ua.chrome)
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    r = requests.get(url, headers=header, allow_redirects=True, cookies=cookies)
    if r.status_code != 200:
        return (False, "", r.status_code)
    html_content = r.text

    # extract full-text link
    (res, full_text_url, _) = _extract_full_text_link(html_content)
    if not res:
        return (res, full_text_url, -1)
    
    # request from full-text link
    (res, full_text, code) = _request_from_full_text_url(full_text_url)
    return (res, full_text, code)

def request_paper(pmid: str):
    pmid = pmid.strip()
    ua = UserAgent()
    print(ua.chrome)
    header = headers
    header["User-Agent"] = str(ua.chrome)
    print(header)

    if pmid.startswith("http://") or pmid.startswith("https://"):
        return _request_from_full_text_url(pmid)
    
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/"
    r = requests.get(url, headers=header, allow_redirects=True, cookies=cookies)
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
    text = soup.get_text(separator="\n", strip=True)
    return text

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


