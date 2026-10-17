"""FastAPI entrypoint. Only a health check for now; /ask comes in step 5."""

import httpx
from fastapi import FastAPI

from eu_reg_rag import __version__
from eu_reg_rag.config import settings

app = FastAPI(title="EU Reg RAG", version=__version__)


@app.get("/health")
def health() -> dict:
    status = {"api": "ok", "ollama": "down", "qdrant": "down"}
    try:
        httpx.get(f"{settings.ollama_url}/api/tags", timeout=2).raise_for_status()
        status["ollama"] = "ok"
    except httpx.HTTPError:
        pass
    if settings.qdrant_mode == "embedded":
        status["qdrant"] = "embedded"
    else:
        try:
            httpx.get(f"{settings.qdrant_url}/healthz", timeout=2).raise_for_status()
            status["qdrant"] = "ok"
        except httpx.HTTPError:
            pass
    return status
