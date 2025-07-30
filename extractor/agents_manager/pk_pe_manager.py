from langchain_openai.chat_models.base import BaseChatOpenAI
import json
import pandas as pd

from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow
from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.utils import convert_html_to_text_no_table, remove_references

class PKPEManager:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def _extract_pmid_info(self, pmid: str):
        pmid_db = PMIDDB()
        pmid_info = pmid_db.select_pmid_info(pmid)
        if pmid_info is not None: # pmid has been saved to database
            return True
        retriever = ArticleRetriever()
        res, html_content, code = retriever.request_article(pmid)
        if not res:
            return False
        extractor = HtmlTableExtractor()
        tables = extractor.extract_tables(html_content)
        sections = extractor.extract_sections(html_content)
        abstract = extractor.extract_abstract(html_content)
        title = extractor.extract_title(html_content)
        full_text = convert_html_to_text_no_table(html_content)
        full_text = remove_references(full_text)
        pmid_db.insert_pmid_info(pmid, title, abstract, full_text, tables, sections)
        return True

    def _run_pk_workflows(self, pmid: str):
        pass

    def run(self, pmid: str):
        workflow = PKSumWorkflow(llm=self.llm)
        workflow.build()
        dfs: list[pd.DataFrame] = []

        
