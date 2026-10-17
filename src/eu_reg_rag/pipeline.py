"""End-to-end RAG pipeline: retrieve (eval-chosen defaults) -> generate with citations."""

from __future__ import annotations

import time
from dataclasses import asdict

from eu_reg_rag.generate import LLMFn, answer_question, ollama_llm
from eu_reg_rag.retrieval import (
    BINDING_KINDS,
    HybridRetriever,
    get_client,
    load_chunks,
    sentence_transformer_encoder,
)

# Chosen by eval/results.md: hybrid, articles-first, 1 chunk per article, no reranker.
RETRIEVAL_DEFAULTS = {"mode": "hybrid", "rerank": False, "kinds": BINDING_KINDS, "max_per_unit": 1}


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLMFn, k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.k = k

    @classmethod
    def from_settings(cls) -> RAGPipeline:
        retriever = HybridRetriever(load_chunks(), sentence_transformer_encoder(), get_client())
        return cls(retriever, ollama_llm())

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        results = self.retriever.search(question, k=self.k, **RETRIEVAL_DEFAULTS)
        t1 = time.perf_counter()
        ans = answer_question(question, [r.chunk for r in results], self.llm)
        t2 = time.perf_counter()
        out = asdict(ans)
        out["sources"] = [
            {"label": c.label, "id": c.chunk_id, "title": c.title} for c in ans.citations
        ]
        out["timing_ms"] = {
            "retrieval": round((t1 - t0) * 1000),
            "generation": round((t2 - t1) * 1000),
        }
        return out
