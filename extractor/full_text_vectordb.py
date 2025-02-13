
from langchain_core.documents import Document
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
import uuid
from pymilvus import (
    MilvusException,
    connections,
    utility,
)
import logging

logger = logging.getLogger(__name__)

EMBEDDINGS_COLLECTION_NAME = "ArticleEmbeddings"

class VectorDBMilvus:
    def __init__(
        self,
        connection_args: dict,
        embedding_func: OpenAIEmbeddings,
        collection_name: str | None = None,
    ):
        self._connection_args = connection_args
        self._embedding_func = embedding_func
        self._collection_name = collection_name or EMBEDDINGS_COLLECTION_NAME
        self._alias = None

    def _create_connection_alias(self):
        alias = uuid.uuid4().hex
        try:
            connections.connect(
                host=self._connection_args['host'],
                port=self._connection_args["port"],
                user=self._connection_args.get("user", ""),
                password=self._connection_args.get("password", ""),
                alias=alias,
            )
            logging.debug(f"Create new connection using: {alias}")
            return alias

        except Exception as e:
            logger.error(e)
            raise e

    def _load_embedding_collection(self):
        try:
            self._collection_embedding = Milvus(
                embedding_function=self._embedding_func,
                collection_name=self._collection_name,
                connection_args=self._connection_args,
            )
        except MilvusException as e:
            logger.error(e)
            raise e
        
    def _insert_data(self, docs: list[str]):
        try:
            self._collection_embedding = Milvus.from_texts(
                texts=docs,
                embedding=self._embedding_func,
                collection_name=self._collection_name,
                connection_args=self._connection_args,
            )            
        except MilvusException as e:
            logger.error(
                "Failed to insert data to embedding collection " f"{self._embedding_name}.",
            )
            raise e

    def connect(self):
        self._alias = self._create_connection_alias()
        self._load_embedding_collection()

    def save_documents(self, docs: list[str]):
        self._insert_data(docs)

    def similarity_search(self, query: str, k: int=3):
        if self._collection_embedding is None:
            logger.error("Please initialize first")
            return None
        
        result = self._collection_embedding.similarity_search(
            query=query,
            k=k,
        )
        return result
    
    def remove_collection(self):
        try:
            from pymilvus import utility
            utility.drop_collection(collection_name=self._collection_name, using=self._alias)
        except MilvusException as e:
            logger.error(f"Failed to remove collection: {self._collection_name}")
            raise e



