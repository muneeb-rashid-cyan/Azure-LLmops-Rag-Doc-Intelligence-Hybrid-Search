"""
cosmos_client.py
----------------
Cosmos DB operations for storing documents, sessions, and evals.
"""

from functools import lru_cache
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from loguru import logger
from config import get_settings


class RAGCosmosClient:
    def __init__(self):
        s = get_settings()
        self._client = CosmosClient(s.cosmos_endpoint, credential=s.cosmos_key)
        self._db     = self._client.get_database_client(s.cosmos_database)
        self._docs   = self._db.get_container_client(s.cosmos_container_docs)
        self._sessions = self._db.get_container_client(s.cosmos_container_sessions)
        self._setup()

    def _setup(self):
        """Create DB and containers if they don't exist."""
        s = get_settings()
        try:
            self._client.create_database_if_not_exists(s.cosmos_database)
            self._db.create_container_if_not_exists(
                id=s.cosmos_container_docs,
                partition_key=PartitionKey(path="/id"),
            )
            self._db.create_container_if_not_exists(
                id=s.cosmos_container_sessions,
                partition_key=PartitionKey(path="/session_id"),
            )
        except Exception as e:
            logger.warning(f"Cosmos setup warning (may already exist): {e}")

    def upsert_document_record(self, doc: dict) -> None:
        self._docs.upsert_item(doc)
        logger.debug(f"Cosmos: upserted document record {doc.get('id')}")

    def list_documents(self) -> list[dict]:
        return list(self._docs.read_all_items())

    def save_session(self, session: dict) -> None:
        self._sessions.upsert_item(session)
        logger.debug(f"Cosmos: saved session {session.get('id')}")

    def list_sessions(self, limit: int = 50) -> list[dict]:
        query = f"SELECT TOP {limit} * FROM c ORDER BY c._ts DESC"
        return list(self._sessions.query_items(query, enable_cross_partition_query=True))


@lru_cache
def get_cosmos_client() -> RAGCosmosClient:
    return RAGCosmosClient()