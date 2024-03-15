from typing import List, Any, Tuple, Dict, Optional
from fake_useragent import UserAgent
from bs4 import BeautifulSoup, Tag
import pandas as pd
import logging
import os
import shortuuid

from src.article_stamper import Stamper
from src.make_request import make_article_request, make_get_request
from src.constants import (
    headers,
    cookies,
    FULL_TEXT_LENGTH_THRESHOLD,
    MAX_FULL_TEXT_LENGTH,
)
from src.utils import decode_url

logger = logging.getLogger(__name__)

class PaperRetriver(object):
    def __init__(self, stamper: Stamper):
        self.stamper = stamper

    def _request_by_curl(self, url:str) -> Tuple[bool, str, int]:
        from subprocess import Popen, PIPE
        args = ["curl", url]
        prc = Popen(args, shell=False, stdout=PIPE, stderr=PIPE)
        stdout, stderr = prc.communicate()
        content = str(stdout)
        soup = BeautifulSoup(content, "html.parser")
        aTag = soup.find("a")
        if aTag is None:
            self.stamper.log_error(f"Can't obtain redirect url from _request_by_url({url})")
            return (False, "Can't obtain redirect url", -1)
        redirect_url = aTag.get("href")
        return self._request_from_full_text_url(redirect_url)
    
    def _request_from_full_text_url(self, full_text_url: str) -> Tuple[bool, str, int]:
        ua = UserAgent()
        header = headers
        header["User-Agent"] = str(ua.chrome)
        print(f"full-text url: {full_text_url}")
        r = make_get_request(full_text_url, headers=header, allow_redirects=True, cookies=cookies)
        if r.status_code != 200:
            # failed to get html content, maybe it only allow redirect access
            if r.status_code != 403: # Unknown reason
                self.stamper.log_error(
                    f"Can't access full-text url from _request_from_full_text_url({full_text_url})"
                )
                return (False, "Can't access full-text url", r.status_code)
            # try to handle redirect url
            return self._request_by_curl(full_text_url)
        
        # successfully get html context, check if there is redirect link in it
        content_length = len(r.text) # check the length of full-text
        soup = BeautifulSoup(r.text, "html.parser")
        tag: Tag = soup.find(id="redirectURL")
        if ( content_length < FULL_TEXT_LENGTH_THRESHOLD and
            tag is not None):
            redirect_url = tag.get("value", None)
            if redirect_url is None:
                return (False, "Can't accesss redirect url", -1)
            redirect_url = decode_url(redirect_url)
            print(f"redirected url is: {redirect_url}")
            r = make_get_request(redirect_url, headers=header, allow_redirects=True, cookies=cookies)
            if r.status_code != 200:
                self.stamper.log_error(
                    f"Failed to access redirect url _request_from_full_text({redirect_url})"
                )
                return (False, "Failed to access redirect url", r.status_code)
            return (True, r.text, r.status_code)
        else:
            return (True, r.text, r.status_code)
    
    def _extract_full_text_link(self, html_content: str) -> Tuple[bool, str, int]:
        soup = BeautifulSoup(html_content, "html.parser")
        tags = soup.select("div.full-view > div.full-text-links-list > a.link-item")
        if tags is None or len(tags) == 0:
            return (False, "Can't get full-text url by selector", -1)
        aTag = tags[0]
        full_text_url = aTag.attrs.get("href", None)
        if full_text_url is None:
            return (False, "Can't get full-text url from href attribute", -1)
        return (True, full_text_url, 200)
    
    def _request_by_abstract_page(self, pmid: str):
        ua = UserAgent()
        header = headers
        header["User-Agent"] = str(ua.chrome)
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        r = make_get_request(url, headers=header, allow_redirects=True, cookies=cookies)
        if r.status_code != 200:
            return (False, "", r.status_code)
        html_content = r.text
    
        # extract full-text link
        (res, full_text_url, _) = self._extract_full_text_link(html_content)
        if not res:
            return (res, full_text_url, -1)
        
        # request from full-text link
        (res, full_text, code) = self._request_from_full_text_url(full_text_url)
        return (res, full_text, code)
    
    def _request_full_text_url(self, url: str, pmid: str):
        fn = shortuuid.uuid()
        folder = os.environ.get("TEMP_FOLDER", "/tmp")
        fn = os.path.join(folder, fn)
        res = make_article_request(url, fn)
        return res.status_code == 200, res.text, res.status_code

    def _request_pmc_full_text(self, pmid: str):
        header = headers
        header["User-Agent"] = str(ua.chrome)
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/"
        res = make_get_request(url, headers=header, allow_redirects=True, cookies=cookies)
        if res.status_code == 200:
            return True, res.text, res.status_code
        return False, res.reason, res.status_code
    
    def request_paper(self, pmid: str):
        pmid = pmid.strip()
        ua = UserAgent()
        print(ua.chrome)
        header = headers
        header["User-Agent"] = str(ua.chrome)
        print(header)
    
        if pmid.startswith("http://") or pmid.startswith("https://"):
            return self._request_from_full_text_url(pmid)
        
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/"
        r = make_get_request(url, headers=header, allow_redirects=True, cookies=cookies)
        if r.status_code == 200:
            return (True, r.text, r.status_code)
        
        # Maybe no full-text in PubMed, in this case, we will try
        # full-text link in abstract page (https://pubmed.ncbi.nlm.nih.gov/{pmid}/)
        print(f"{r.status_code}: {r.reason}")
        logger.warn(f"{r.status_code}: {r.reason}")
        (res, text, code) = self._request_by_abstract_page(pmid)
    
        self.stamper.output_html(text) 
        return (res, text, code)
    
    def convert_html_to_text(self, html_content: str, exclude_tables: Optional[bool] = False) -> str:
        '''
        This function is used to convert html string to text, that is,
        extract text from html content, including tables.
        '''
        soup = BeautifulSoup(html_content, "html.parser")
        if exclude_tables:
            results = soup.find_all("div", attrs={"class": "xtable"})
            for result in results:
                result.clear()
        text = soup.get_text(separator="\n", strip=True)
        return text
    
    def extract_tables_from_html(self, html_content: str) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html_content, "html.parser")
        tags = soup.select("div.table-wrap.anchored.whole_rhythm")
        tables = []
        for tag in tags:
            tbl_soup = BeautifulSoup(str(tag), "html.parser")
            caption = tbl_soup.select("div.caption")
            caption = caption[0].text if len(caption) > 0 else ""
            table = tbl_soup.select("div.xtable")
            table = str(table[0]) if len(table) > 0 else ""
            table = self.convert_table_to_dataframe(table)
            footnote = tbl_soup.select("div.tblwrap-foot")
            footnote = footnote[0].text if len(footnote) > 0 else ""
            tables.append({"caption": caption, "table": table, "footnote": footnote})
    
        return tables
    
    def convert_table_to_dataframe(self, table: str):
        try:
            df = pd.read_html(table)
            return df[0]
        except Exception as e:
            logger.error(e)
            print(e)
            return None

    def remove_references(self, text: str):
        ix = text.lower().rfind("references")
        if ix < 0:
            logger.warn(f"Can't find 'References' in paper {self.stamper.pmid}")
            return text
        return text[:ix]

class FakePaperRetriver(PaperRetriver):
    def __init__(self, stamper: Stamper):
        super().__init__(stamper)
    
    def request_paper(self, pmid: str):
        pmid_folder = f"./tmp/{pmid}"
        if not os.path.exists(pmid_folder):
            return super().request_paper(pmid)
        for root, dirs, files in os.walk(pmid_folder):
            html_files = [f for f in files if f.endswith("html")]
            html_files.sort()
            break
        if len(html_files) == 0:
            return super().request_paper(pmid)
        the_file = os.path.join(root, html_files[-1])
        with open(the_file, "r") as fobj:
            content = fobj.read()
            return True, content, 200

        

def test_request_paper_and_obtain_tables():
    pmid = "23106931"
    stamper = Stamper()
    retriver = PaperRetriver(stamper)
    (res, content, status_code) = retriver.request_paper(pmid)
    if res:
        tables = retriver.extract_tables_from_html(content)
        for tbl in tables:
            print(tbl["caption"])
            df = retriver.convert_table_to_dataframe(tbl["table"])
            print(df)
            print(tbl["footnote"])

def test_request_paper_by_full_text_link():
    # pmid = "30322724"
    # pmid = "24785239"
    pmid = "30284023"
    stamper = Stamper()
    retriever = PaperRetriver(stamper)
    (res, content, status_code) = retriever.request_paper(pmid)
    text = retriever.convert_html_to_text(content)
    ix = text.find("References")
    content = text[:ix]
    assert res



if __name__ == "__main__":
    test_request_paper_by_full_text_link()


