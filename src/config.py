import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def build_database_url_from_postgres_env(env: dict[str, str] | None = None) -> str:
    """Build SQLAlchemy PostgreSQL URL from POSTGRES_* env vars (password URL-encoded)."""
    e = env if env is not None else os.environ
    host = e.get("POSTGRES_HOST", "localhost")
    port = e.get("POSTGRES_PORT", "5432")
    user = e.get("POSTGRES_USER", "postgres")
    password = e.get("POSTGRES_PASSWORD", "postgres")
    database = e.get("POSTGRES_DATABASE") or e.get("POSTGRES_DB", "finrag")
    u = quote_plus(user)
    p = quote_plus(password)
    return f"postgresql://{u}:{p}@{host}:{port}/{database}"


def get_database_url() -> str:
    """
    Resolve database URL: optional explicit DATABASE_URL wins, else POSTGRES_* composition.
    """
    raw = os.environ.get("DATABASE_URL")
    if raw and raw.strip():
        return raw.strip()
    return build_database_url_from_postgres_env()


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _as_json_dict(value: str | None, default: dict[str, Any]) -> dict[str, Any]:
    if value is None or value.strip() == "":
        return default
    try:
        data = json.loads(value)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return default


def _as_optional(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def get_config() -> dict[str, Any]:
    env = os.environ
    persist_directory = env.get("PERSIST_DIRECTORY", "./data")
    return {
        "database_url": get_database_url(),
        "postgres_max_connections": _as_int(env.get("POSTGRES_MAX_CONNECTIONS"), 12),
        "chroma_server_host": _as_optional(env.get("CHROMA_SERVER_HOST", "localhost")),
        "chroma_server_port": _as_int(env.get("CHROMA_SERVER_PORT"), 8000),
        "default_collection_top_k": _as_int(env.get("DEFAULT_COLLECTION_TOP_K"), 10),
        "collections_top_k": _as_json_dict(env.get("COLLECTIONS_TOP_K"), {}),
        "persist_directory": persist_directory,
        "bm25_storage_backend": env.get("BM25_STORAGE_BACKEND", "local"),
        "bm25_local_base_dir": _as_optional(env.get("BM25_LOCAL_BASE_DIR")),
        "bm25_s3_bucket": _as_optional(env.get("BM25_S3_BUCKET")),
        "bm25_s3_prefix": _as_optional(env.get("BM25_S3_PREFIX")),
        "bm25_s3_region": _as_optional(env.get("BM25_S3_REGION")),
        "embeddings_model_name": env.get("EMBEDDINGS_MODEL_NAME", "BAAI/bge-m3"),
        "llm_model_name": env.get("LLM_MODEL_NAME", "Qwen/Qwen2___5-72B-Instruct-AWQ"),
        "chat_llm_model_name": _as_optional(env.get("CHAT_LLM_MODEL_NAME")),
        "llm_service_base_url": env.get("LLM_SERVICE_BASE_URL", "http://127.0.0.1:8001"),
        "llm_service_llm_endpoint_path": env.get("LLM_SERVICE_LLM_ENDPOINT_PATH", "/llm/complete"),
        "llm_service_timeout_seconds": _as_int(env.get("LLM_SERVICE_TIMEOUT_SECONDS"), 120),
        "llm_service_provider": env.get("LLM_SERVICE_PROVIDER", "openai"),
        "llm_service_embeddings_provider": _as_optional(env.get("LLM_SERVICE_EMBEDDINGS_PROVIDER")),
        "llm_service_embeddings_batch_size": _as_int(env.get("LLM_SERVICE_EMBEDDINGS_BATCH_SIZE"), 64),
        "rerank_model": env.get("RERANK_MODEL", "BAAI/bge-reranker-v2-gemma"),
        "rerank_topk": _as_int(env.get("RERANK_TOPK"), 5),
        "log_level": env.get("LOG_LEVEL", "INFO"),
        "bearer_token": _as_optional(env.get("BEARER_TOKEN")),
        "feedback_processing_enabled": _as_bool(env.get("FEEDBACK_PROCESSING_ENABLED"), False),
        "feedback_processing_interval_seconds": _as_int(env.get("FEEDBACK_PROCESSING_INTERVAL_SECONDS"), 600),
        "feedback_last_processed_id_file": env.get("FEEDBACK_LAST_PROCESSED_ID_FILE", "log/feedback_last_processed_id.txt"),
        "feedback_categories_path": env.get("FEEDBACK_CATEGORIES_PATH", "config/feedback_categories.json"),
        "feedback_classifier_provider": _as_optional(env.get("FEEDBACK_CLASSIFIER_PROVIDER")),
        "feedback_classifier_model": env.get("FEEDBACK_CLASSIFIER_MODEL", "deepseek-v3"),
        "treerag_enabled": _as_bool(env.get("TREERAG_ENABLED"), False),
        "treerag_max_depth": _as_int(env.get("TREERAG_MAX_DEPTH"), 2),
        "treerag_branching_factor": _as_int(env.get("TREERAG_BRANCHING_FACTOR"), 3),
        "treerag_max_workers": _as_int(env.get("TREERAG_MAX_WORKERS"), 6),
        "treerag_use_hyde": _as_bool(env.get("TREERAG_USE_HYDE"), False),
        "treerag_retrieve_max_chunks": _as_int(env.get("TREERAG_RETRIEVE_MAX_CHUNKS"), 40),
        "treerag_session_ttl_seconds": _as_int(env.get("TREERAG_SESSION_TTL_SECONDS"), 1800),
        "treerag_llm_provider": _as_optional(env.get("TREERAG_LLM_PROVIDER")),
        "treerag_llm_model_name": _as_optional(env.get("TREERAG_LLM_MODEL_NAME")),
        "treerag_planner_model": _as_optional(env.get("TREERAG_PLANNER_MODEL")),
        "treerag_answer_model": _as_optional(env.get("TREERAG_ANSWER_MODEL")),
        "qa_table_persist_directory": env.get("QA_TABLE_PERSIST_DIRECTORY", str(Path(persist_directory) / "qa_chroma")),
        "qa_table_directory": _as_optional(env.get("QA_TABLE_DIRECTORY")),
        "frequent_qa_directory": _as_optional(env.get("FREQUENT_QA_DIRECTORY")),
        "test_llm_model_name": env.get("TEST_LLM_MODEL_NAME", "llama3"),
        "test_llm_api_key": env.get("TEST_LLM_API_KEY", "EMPTY"),
        "test_llm_base_url": env.get("TEST_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        "port": _as_int(env.get("PORT"), 6005),
    }
