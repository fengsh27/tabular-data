from __future__ import annotations
import os
from dataclasses import dataclass
import logging
from typing import Dict, Optional, Union
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FullTextResult:
    pmid: str
    url: str
    code: int = 200
    pmcid: Optional[str] = None
    content_type: Optional[str] = None
    content: Optional[Union[str, bytes]] = None
    text_and_tables: Optional[str] = None


class PubMedFullTextRetriever:
    """Retrieve PMC full text for PubMed papers (HTML preferred, PDF fallback)."""

    def __init__(
        self,
        email: Optional[str] = None,
        tool: str = "biomarker_curator",
        api_key: Optional[str] = None,
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.email = email or os.getenv("NCBI_EMAIL")
        self.tool = tool
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(self._default_headers())

    def retrieve(self, pmid: str, prefer: str = "html", fallback: bool = True) -> FullTextResult:
        """Return full text when available. Prefer PMC HTML/PDF, fallback to OA sources."""
        pmcid = self._find_pmcid(pmid)
        if not pmcid and fallback:
            fallback_result = self._fetch_unpaywall(pmid)
            if fallback_result is not None:
                return fallback_result
            raise ValueError(
                f"No PMC full text available for PMID {pmid}. "
                "Provide an email to enable Unpaywall fallback."
            )

        prefer = prefer.lower()
        if prefer not in {"html", "pdf"}:
            raise ValueError("prefer must be 'html' or 'pdf'.")

        if prefer == "html":
            html_result = self._fetch_html(pmid, pmcid)
            if html_result.code < 400:
                return html_result
            logger.warning(
                "HTML fetch failed for %s (%s): %s",
                pmid,
                pmcid,
                html_result.code,
            )
            if not fallback:
                raise ValueError(
                    f"HTML fetch failed for {pmid} ({pmcid}): {html_result.code}. "
                    "Provide an email to enable Unpaywall fallback."
                )
            pdf_result = self._fetch_pdf(pmid, pmcid)
            if pdf_result.code < 400:
                return pdf_result
            logger.warning(
                "PDF fetch failed for %s (%s): %s",
                pmid,
                pmcid,
                pdf_result.code,
            )
            return self._fetch_xml(pmid, pmcid)

        pdf_result = self._fetch_pdf(pmid, pmcid)
        if pdf_result.code < 400:
            return pdf_result
        logger.warning("PDF fetch failed for %s (%s): %s", pmid, pmcid, pdf_result.code)
        return self._fetch_xml(pmid, pmcid)

    def _fetch_html(self, pmid: str, pmcid: str) -> FullTextResult:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        response = self.session.get(url, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTML fetch failed for %s (%s): %s", pmid, pmcid, exc)
            return FullTextResult(
                pmid=pmid,
                pmcid=pmcid,
                content_type="text/html",
                content=response.content,
                url=url,
                code=response.status_code,
            )
        logger.info("PubMedFullTextRetriever: fetched HTML for %s", pmcid)
        content = response.content.decode("utf-8")
        return FullTextResult(
            pmid=pmid,
            pmcid=pmcid,
            content_type="text/html",
            content=content,
            url=url,
            code=response.status_code,
        )

    def _fetch_pdf(self, pmid: str, pmcid: str) -> FullTextResult:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        response = self.session.get(url, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.warning("PDF fetch failed for %s (%s): %s", pmid, pmcid, exc)
            return FullTextResult(
                pmid=pmid,
                pmcid=pmcid,
                content_type="application/pdf",
                content=response.content,
                url=url,
                code=response.status_code,
            )
        logger.info("PubMedFullTextRetriever: fetched PDF for %s", pmcid)
        return FullTextResult(
            pmid=pmid,
            pmcid=pmcid,
            content_type="application/pdf",
            content=response.content,
            url=url,
            code=response.status_code,
        )

    def _fetch_xml(self, pmid: str, pmcid: str) -> FullTextResult:
        params: Dict[str, str] = {
            "db": "pmc",
            "id": pmcid,
            "rettype": "full",
            "retmode": "xml",
        }
        self._add_common_params(params)
        url = f"{self.base_url}/efetch.fcgi"
        response = self.session.get(url, params=params, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.warning("XML fetch failed for %s (%s): %s", pmid, pmcid, exc)
            return FullTextResult(
                pmid=pmid,
                pmcid=pmcid,
                content_type="application/xml",
                content=response.content,
                url=str(response.url),
                code=response.status_code,
            )
        logger.info("PubMedFullTextRetriever: fetched XML for %s", pmcid)
        return FullTextResult(
            pmid=pmid,
            pmcid=pmcid,
            content_type="application/xml",
            content=response.content,
            url=str(response.url),
            code=response.status_code,
        )

    def _fetch_html_with_article_retriever(self, pmid: str) -> FullTextResult:
        from biomarker_curator.utils.article_retriever import ArticleRetriever

        retriever = ArticleRetriever()
        res, html_content, code = retriever.request_article(pmid)
        return FullTextResult(
            pmid=pmid,
            content_type="text/html",
            content=html_content,
            url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/",
            code=code,
        )

    def _fetch_unpaywall(self, pmid: str) -> Optional[FullTextResult]:
        if not self.email:
            logger.warning("Unpaywall lookup skipped for %s: email not configured", pmid)
            return None
        doi = self._find_doi(pmid)
        if not doi:
            logger.warning("Unpaywall lookup skipped for %s: DOI not found", pmid)
            return None
        api_url = f"https://api.unpaywall.org/v2/{doi}"
        response = self.session.get(api_url, params={"email": self.email}, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.warning("Unpaywall lookup failed for %s (%s): %s", pmid, doi, exc)
            return self._fetch_html_with_article_retriever(pmid)
        payload = response.json()
        location = payload.get("best_oa_location") or {}
        fulltext_url = location.get("url_for_pdf") or location.get("url")
        if not fulltext_url:
            logger.warning("Unpaywall lookup had no OA URL for %s (%s)", pmid, doi)
            return self._fetch_html_with_article_retriever(pmid)
        fulltext_response = self.session.get(fulltext_url, timeout=self.timeout)
        try:
            fulltext_response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.warning("Unpaywall full text fetch failed for %s (%s): %s", pmid, doi, exc)
            return self._fetch_html_with_article_retriever(pmid)
            
        logger.info("PubMedFullTextRetriever: fetched OA full text for %s", pmid)
        return FullTextResult(
            pmid=pmid,
            pmcid=None,
            content_type=self._content_type_for_url(fulltext_url, fulltext_response),
            content=fulltext_response.content,
            url=fulltext_url,
            code=fulltext_response.status_code,
        )

    def _find_pmcid(self, pmid: str) -> Optional[str]:
        params: Dict[str, str] = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "xml",
        }
        self._add_common_params(params)
        response = self.session.get(
            f"{self.base_url}/elink.fcgi",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_pmcid(response.text)

    def _find_doi(self, pmid: str) -> Optional[str]:
        params: Dict[str, str] = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
        }
        self._add_common_params(params)
        response = self.session.get(
            f"{self.base_url}/efetch.fcgi",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_doi(response.text)

    @staticmethod
    def _parse_pmcid(xml_text: str) -> Optional[str]:
        root = ET.fromstring(xml_text)
        for linkset in root.findall(".//LinkSetDb"):
            link_name = linkset.findtext("LinkName")
            if link_name not in {"pubmed_pmc", "pubmed_pmc_local"}:
                continue
            id_nodes = linkset.findall("./Link/Id")
            if not id_nodes:
                continue
            pmc_id = id_nodes[0].text
            if not pmc_id:
                continue
            return f"PMC{pmc_id}" if not pmc_id.startswith("PMC") else pmc_id
        return None

    @staticmethod
    def _parse_doi(xml_text: str) -> Optional[str]:
        root = ET.fromstring(xml_text)
        doi_node = root.find(".//ArticleIdList/ArticleId[@IdType='doi']")
        if doi_node is None or doi_node.text is None:
            return None
        return doi_node.text.strip()

    def _add_common_params(self, params: Dict[str, str]) -> None:
        if self.email:
            params["email"] = self.email
        if self.tool:
            params["tool"] = self.tool
        if self.api_key:
            params["api_key"] = self.api_key

    def _default_headers(self) -> Dict[str, str]:
        contact = self.email or "unknown"
        user_agent = f"{self.tool} ({contact})"
        return {
            "User-Agent": user_agent,
            "Accept": "text/html,application/pdf,application/xml;q=0.9,*/*;q=0.8",
        }

    @staticmethod
    def _content_type_for_url(url: str, response: requests.Response) -> str:
        content_type = response.headers.get("Content-Type")
        if content_type:
            return content_type.split(";")[0].strip()
        if url.lower().endswith(".pdf"):
            return "application/pdf"
        if url.lower().endswith(".xml"):
            return "application/xml"
        if url.lower().endswith(".html") or url.lower().endswith(".htm"):
            return "text/html"
        return "application/octet-stream"
