"""FastAPI app: /health and /ask.

Models load once at startup (lifespan), not per request.
Note: embedded Qdrant locks its folder, so stop the API before running index/search CLIs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from eu_reg_rag import __version__
from eu_reg_rag.config import settings

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from eu_reg_rag.pipeline import RAGPipeline

    try:
        state["pipeline"] = RAGPipeline.from_settings()
    except SystemExit as e:  # missing index/chunks: keep /health up, report on /ask
        state["error"] = str(e)
    yield
    state.clear()


app = FastAPI(title="EU Reg RAG", version=__version__, lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health() -> dict:
    status = {"api": "ok", "ollama": "down", "qdrant": "down", "pipeline": "down"}
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
    if "pipeline" in state:
        status["pipeline"] = "ok"
    return status


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if "pipeline" not in state:
        raise HTTPException(503, state.get("error", "pipeline not loaded"))
    try:
        return state["pipeline"].ask(req.question)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"LLM backend error: {e}") from e
