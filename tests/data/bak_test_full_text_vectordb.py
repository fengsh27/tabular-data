
import pytest
import re
import os

from extractor.full_text_vectordb import VectorDBMilvus
from extractor.html_table_extractor import HtmlTableExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def remove_html_tags(html: str) -> str:
    return re.sub(r"<.*?>", " ", html)

def clean_text(content: str) -> str:
    content = content.replace('\n', ' ')
    return re.sub(r"\s+", ' ', content).strip()

def test_VectorMilvus():
    with open("./tests/30950674.html", "r") as fobj:
        html = fobj.read()
        table_extractor = HtmlTableExtractor()
        tables, html_new = table_extractor.extract_tables_and_remove_from_html(html)
                
        content = remove_html_tags(html_new)
        content = clean_text(content)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separators=[" ", ",", ".", "\n", ";"],
        )
        # content_bytes = content.encode(encoding="utf-8")
        splitted_texts = splitter.split_text(content)

        embeddings = AzureOpenAIEmbeddings(
            api_key=os.environ.get("OPENAI_4O_API_KEY", ""),
            azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", ""),
            azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", ""),
            model=os.environ.get("AZURE_OPENAI_EMBEDDINGS_MODEL", ""),
        )
        db = VectorDBMilvus(
            connection_args={
                'host': "10.95.224.94",
                'port': '19530',
            },
            collection_name="test_1",
            embedding_func=embeddings,
        )
        db.connect()
        db.save_documents(splitted_texts)

        result = db.similarity_search(tables[1]["caption"])
        assert len(result) == 3

        db.remove_collection()

        



