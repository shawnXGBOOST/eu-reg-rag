"""FastAPI app: /health and /ask.

Models load once at startup (lifespan), not per request.
Note: embedded Qdrant locks its folder, so stop the API before running index/search CLIs.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from eu_reg_rag import __version__
from eu_reg_rag.config import settings

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from eu_reg_rag.agent import RAGAgent

    try:
        state["pipeline"] = RAGAgent.from_settings(settings.agent_strategy)
    except SystemExit as e:  # missing index/chunks: keep /health up, report on /ask
        state["error"] = str(e)
    yield
    state.clear()


app = FastAPI(title="EU Reg RAG", version=__version__, lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


UI_FILE = Path(__file__).parent / "static" / "index.html"

_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    """Per-IP cap on /ask requests (settings.rate_limit_per_hour; 0 = off). In memory, so it
    resets on restart. Behind a proxy (Hugging Face Spaces) the client IP is in X-Forwarded-For."""
    limit = settings.rate_limit_per_hour
    if not limit:
        return
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, f"Demo limit reached ({limit} questions per hour). Try later.")
    q.append(now)


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(UI_FILE)


@app.get("/health")
def health() -> dict:
    status = {"api": "ok", "ollama": "down", "qdrant": "down", "pipeline": "down"}
    if settings.llm_backend == "openai":  # hosted LLM: don't spend a request just to ping it
        status["ollama"] = "n/a"
        status["llm"] = "configured" if (settings.llm_api_key or settings.hf_token) else "no key"
    else:
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
def ask(req: AskRequest, request: Request) -> dict:
    rate_limit(request)
    if "pipeline" not in state:
        raise HTTPException(503, state.get("error", "pipeline not loaded"))
    try:
        return state["pipeline"].ask(req.question)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"LLM backend error: {e}") from e


@app.post("/ask/stream")
def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
    """NDJSON stream: one JSON event per line (sources, rewrite, token, attempt_done, final)."""
    rate_limit(request)
    if "pipeline" not in state:
        raise HTTPException(503, state.get("error", "pipeline not loaded"))

    def events():
        try:
            for event in state["pipeline"].ask_stream(req.question):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except httpx.HTTPError as e:
            yield json.dumps({"type": "error", "message": f"LLM backend error: {e}"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
