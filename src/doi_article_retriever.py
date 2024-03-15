from typing import Tuple
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import logging

from src.make_request import make_get_request
from src.utils import (
    request_from_full_text_url
)

logger = logging.getLogger(__name__)

class AbstractRetriever(ABC):
    def __init__(self, abstract_page_html: str):
        self.html_content = abstract_page_html

    @abstractmethod
    def is_applicable(self) -> bool:
        return False
    
    @abstractmethod
    def extract(self, abstract_page_html: str) -> Tuple[bool, str, int]:
        return False, "", 404
    
class DOIRetriever(AbstractRetriever):
    def __init__(self, abstract_page_html: str):
        super().__init__(abstract_page_html)
        self.soup = BeautifulSoup(self.html_content)

    def is_applicable(self) -> bool:
        tags = self.soup.select('ul#full-view-identifiers span.doi > a.id-link')
        if tags is None:
            return False
        return len(tags) > 0
    
    def extract(self) -> Tuple[bool | str | int]:
        res, doi_url = self._extract_doi_url()
        if not res:
            return False, "Failed to extract doi url", -1
        try:
            return self._access_doi_url(doi_url)
        except Exception as e:
            logger.error(e)
            return False, str(e), -1

    def _extract_doi_url(self) -> Tuple[bool, str]:
        tags = self.soup.select('ul#full-view-identifiers span.doi > a.id-link')
        assert tags is not None and len(tags) > 0
        aTag = tags[0]
        full_text_url = aTag.attrs.get("href", None)
        return (full_text_url is not None, full_text_url)
    
    def _access_doi_url(self, url: str) -> Tuple[bool, str, int]:
        return request_from_full_text_url(url)

