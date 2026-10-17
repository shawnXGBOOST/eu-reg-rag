"""Central config. Every setting can be overridden via env vars or a .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (Ollama)
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    llm_temperature: float = 0.1
    # Agent: "fallback" rewrites only after a refusal; "rewrite_first" always rewrites
    agent_strategy: str = "fallback"

    # Embeddings / reranker
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"

    # Vector store: "embedded" = local folder, no Docker; "server" = qdrant_url
    qdrant_mode: str = "embedded"
    qdrant_path: str = "data/qdrant"
    qdrant_url: str = "http://localhost:6333"
    collection: str = "eu_regs"

    # Retrieval
    top_k_retrieve: int = 20  # candidates from EACH of BM25 and dense
    rerank_pool: int = 30  # fused candidates passed to the cross-encoder
    top_k_final: int = 5

    # Paths
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


settings = Settings()
