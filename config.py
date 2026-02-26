"""
config.py
---------
Single source of truth for all settings.
Reads from .env via pydantic-settings.
Import this everywhere — never read os.environ directly.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Azure OpenAI ─────────────────────────────────────────
    azure_openai_endpoint: str
    azure_openai_key: str
    azure_openai_deployment: str = "gpt-4.1"
    azure_openai_emb_deployment: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-08-01-preview"
    azure_openai_region: str = "eastus"

    # ── Document Intelligence ─────────────────────────────────
    document_intelligence_endpoint: str
    document_intelligence_key: str

    # ── Azure AI Search ───────────────────────────────────────
    azure_search_endpoint: str
    azure_search_key: str
    azure_search_index: str = "rag-documents"

    # ── Cosmos DB ─────────────────────────────────────────────
    cosmos_endpoint: str
    cosmos_key: str
    cosmos_database: str = "MultiModalRAG"
    cosmos_container_docs: str = "Documents"
    cosmos_container_sessions: str = "ChatSessions"
    cosmos_container_evals: str = "Evaluations"

    # ── ACR + Web App ─────────────────────────────────────────
    acr_name: str = "costforecastingmlopsacr"
    webapp_name: str = "enhanced-mlops-api-muneeb"
    resource_group: str = "mirlin-ml-dev"

    # ── RAG Pipeline ──────────────────────────────────────────
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    query_rewrite_count: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()