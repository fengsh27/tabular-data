from pathlib import Path
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import logging
import os
import shortuuid

from extractor.make_request import make_article_request, make_get_request
from extractor.constants import (
    headers,
    cookies,
)

logger = logging.getLogger(__name__)

def article_processor(func):
    def wrapper(*args, **kwargs):
        res, text, code = func(*args, **kwargs)
        if res and text is not None and len(text) > 0:
            text = text.replace('\u2009', ' ')
        return res, text, code
    return wrapper

class ArticleRetriever(object):
    def __init__(self):
        pass

    def _request_full_text_from_url(self, url: str):
        """
        request full-text by url
        """
        # img_fn = shortuuid.uuid()
        fn = shortuuid.uuid()
        folder = os.environ.get("TEMP_FOLDER", "./tmp")
        fn = os.path.join(folder, fn)
        # img_fn = os.path.join(folder, img_fn)
        # res = make_article_request(url, fn, img_fn) # capture image
        res = make_article_request(url, fn)
        if res.status_code == 200 and os.path.exists(fn):
            # try to use make_request (it doesn't work to make_request(final_url))
            # res_data = res.json()
            # final_url = res_data.get("final_url", None)
            fobj = open(fn, "r")
            text = fobj.read()
            fobj.close()
            os.unlink(fn)
            return True, text, 200
        return (
            False,
            res.text
            if res.status_code != 200
            else f"failed to request full-text article (temporary file does not exist) - {res.reason}",
            res.status_code,
        )

    def _request_pmc_full_text(self, pmid: str):
        """
        request pmc full-text, its url is:
        https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/
        or (for pmc id)
        https://www.ncbi.nlm.nih.gov/pmc/articles/{pmid}
        """
        if pmid.upper().startswith("PMC"):
            pmid = pmid.upper()
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmid}"
        else:
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/"
        header = headers
        ua = UserAgent()
        header["User-Agent"] = str(ua.chrome)
        res = make_get_request(
            url, headers=header, allow_redirects=True, cookies=cookies
        )
        if res.status_code == 200:
            return True, res.text, res.status_code
        return False, res.reason, res.status_code

    def _extract_full_text_link(self, html_content: str) -> tuple[bool, str, int]:
        """
        extract full-text link from html content
        """
        soup = BeautifulSoup(html_content, "html.parser")
        tags = soup.select("div.full-view > div.full-text-links-list > a.link-item")
        if tags is None or len(tags) == 0:
            return (False, "Can't get full-text url by selector", -1)
        aTag = tags[0]
        full_text_url = aTag.attrs.get("href", None)
        if full_text_url is None:
            return (False, "Can't get full-text url from href attribute", -1)
        return (True, full_text_url, 200)

    def _extract_full_text_url_from_abstract_page(self, pmid: str):
        """
        extract full-text url from pmc abstract page (https://pubmed.ncbi.nlm.nih.gov/{pmid}/)
        """
        ua = UserAgent()
        header = headers
        header["User-Agent"] = str(ua.chrome)
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        r = make_get_request(url, headers=header, allow_redirects=True, cookies=cookies)
        if r.status_code != 200:
            return (False, "", r.status_code)
        html_content = r.text

        # extract full-text link
        return self._extract_full_text_link(html_content)

    @article_processor
    def request_article(self, pmid: str):
        pmid = pmid.strip()

        # support full-text url directly
        if pmid.startswith("http"):
            return self._request_full_text_from_url(pmid)

        pmid_cache = os.environ.get("PMID_CACHE", "false")
        if pmid_cache.lower() == "true":
            data_folder = os.environ.get("DATA_FOLDER", "./data")
            pmid_path = Path(data_folder) / f"{pmid}.html"
            if pmid_path.exists():
                logger.info(f"Found {pmid} paper from cache")
                return True, pmid_path.read_text(), 200

        res, pmc_article, code = self._request_pmc_full_text(pmid)
        if res:
            return True, pmc_article, code
        res, full_text_url, code = self._extract_full_text_url_from_abstract_page(pmid)
        if not res:
            logger.error("Can't extract full-text url from abstract page")
            return res, full_text_url, code
        return self._request_full_text_from_url(full_text_url)


class ExtendArticleRetriever(ArticleRetriever):
    """
    Comparing to ArticleRetriever, ExtendArticleRetriever will check if the article already exists first,
    if yes, the existed article will be returned, otherwise, it will download the article.
    """

    def __init__(self):
        super().__init__()

    def request_article(self, pmid: str):
        pmid_folder = os.environ.get("TEMP_FOLDER", "./tmp")
        pmid_folder = os.path.join(pmid_folder, pmid)
        if not os.path.exists(pmid_folder):
            return super().request_article(pmid)
        html_files = []
        for root, dirs, files in os.walk(pmid_folder):
            html_files = [f for f in files if f.endswith("html")]
            html_files.sort()
            break
        if len(html_files) == 0:
            return super().request_article(pmid)
        the_file = os.path.join(root, html_files[-1])
        with open(the_file, "r") as fobj:
            content = fobj.read()
            return True, content, 200
