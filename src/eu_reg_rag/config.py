"""Central config. Every setting can be overridden via env vars or a .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM backend: "ollama" (local, default) or "openai" (any OpenAI-compatible chat API, used for
    # the hosted demo: Hugging Face router, Groq, OpenRouter, ...)
    llm_backend: str = "ollama"
    llm_base_url: str = "https://router.huggingface.co/v1"
    llm_api_key: str = ""  # falls back to HF_TOKEN
    hf_token: str = ""
    llm_extra_body: str = ""  # optional JSON merged into each request, e.g. provider options
    rate_limit_per_hour: int = 0  # per-IP cap on /ask requests; 0 = off

    # Ollama
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"  # for the "openai" backend use e.g. "Qwen/Qwen3-8B"
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
