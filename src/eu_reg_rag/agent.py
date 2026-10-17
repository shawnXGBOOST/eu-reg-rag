"""Agentic RAG with LangGraph: query rewriting, multi-query retrieval and retry-on-refusal.

    START -> [rewrite]* -> retrieve -> generate -> supported? -> END
                 ^                                   | no (attempts left)
                 +-----------------------------------+
    * strategy "rewrite_first" starts at rewrite; "fallback" starts at retrieve
      and only rewrites after a refusal.

Why: legal text uses its own vocabulary ("deployer", "automated individual decision-making",
"right to compensation"). Users don't. The rewriter bridges that gap; the retry loop gives a
second chance before answering "not covered".
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import asdict
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from eu_reg_rag.generate import Answer, LLMFn, answer_question, cite_label, ollama_llm
from eu_reg_rag.retrieval import (
    BINDING_KINDS,
    HybridRetriever,
    get_client,
    load_chunks,
    rrf_fuse,
    sentence_transformer_encoder,
)

REWRITE_SYSTEM = """You convert user questions about the EU AI Act and the GDPR into search
queries written in the regulations' own legal terminology.
Return JSON only: {"queries": ["...", "..."]} with 1 to 3 short queries.
- Translate everyday words into legal terms, for example: a company using an AI tool -> deployer;
  sue for money -> right to compensation; decided by an algorithm without a human ->
  automated individual decision-making; tech company building AI -> provider.
- If the question has several parts or concerns both regulations, write one query per part.
- Do not answer the question."""

MAX_REWRITES = 3
UNIT_KEYS = ("regulation", "kind", "number")


# ---------- building blocks (usable without the graph, e.g. by the eval) ----------


def parse_queries(raw: str) -> list[str]:
    """Robustly extract queries from the model's JSON (or JSON-ish) output."""
    try:
        data = json.loads(raw)
        items = data.get("queries", []) if isinstance(data, dict) else data
    except (json.JSONDecodeError, AttributeError):
        items = re.findall(r'"([^"]{3,200})"', raw)
        items = [q for q in items if q != "queries"]
    out: list[str] = []
    for q in items if isinstance(items, list) else []:
        q = str(q).strip()
        if q and q.lower() not in {x.lower() for x in out}:
            out.append(q)
    return out[:MAX_REWRITES]


def rewrite_queries(llm: LLMFn, question: str, avoid: list[str] | None = None) -> list[str]:
    user = f"Question: {question}"
    if avoid:
        tried = "; ".join(avoid)
        user += f"\nThese searches already failed, use different legal terms: {tried}"
    return parse_queries(llm(REWRITE_SYSTEM, user))


def multi_query_search(
    retriever: HybridRetriever,
    queries: list[str],
    k: int = 5,
    per_query_k: int = 10,
    per_unit: int = 1,
) -> list[dict]:
    """Search each query (eval-chosen defaults), fuse rankings with RRF, return k articles.

    per_unit=1 keeps one chunk per article (best for retrieval metrics). The agent uses 2 for
    generation: long articles (Art. 4 Definitions) split into chunks, and the best-ranked chunk
    is not always the one holding the answer.
    """
    by_id: dict[str, dict] = {}
    rankings: list[list[str]] = []
    for q in queries:
        results = retriever.search(
            q, k=per_query_k * per_unit, rerank=False, kinds=BINDING_KINDS, max_per_unit=per_unit
        )
        rankings.append([r.chunk["id"] for r in results])
        for r in results:
            by_id.setdefault(r.chunk["id"], r.chunk)
    fused = rrf_fuse(rankings)
    out, counts = [], {}
    for cid in sorted(fused, key=fused.get, reverse=True):
        unit = tuple(by_id[cid][key] for key in UNIT_KEYS)
        if unit not in counts and len(counts) == k:
            continue  # k distinct articles reached; only fill remaining chunks of those
        if counts.get(unit, 0) < per_unit:
            counts[unit] = counts.get(unit, 0) + 1
            out.append(by_id[cid])
    return out


# ---------- graph ----------


class AgentState(TypedDict, total=False):
    question: str
    queries: list[str]
    tried: list[str]
    chunks: list[dict]
    answer: Answer
    attempts: int
    trace: list[dict]


class RAGAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMFn,
        rewrite_llm: LLMFn | None = None,
        strategy: str = "fallback",
        max_attempts: int = 2,
        k: int = 5,
    ):
        if strategy not in ("fallback", "rewrite_first"):
            raise ValueError("strategy must be 'fallback' or 'rewrite_first'")
        self.retriever, self.llm = retriever, llm
        self.rewrite_llm = rewrite_llm or llm
        self.strategy, self.max_attempts, self.k = strategy, max_attempts, k
        self._streaming = False
        self.graph = self._build()

    @classmethod
    def from_settings(cls, strategy: str = "fallback") -> RAGAgent:
        retriever = HybridRetriever(load_chunks(), sentence_transformer_encoder(), get_client())
        return cls(retriever, ollama_llm(), ollama_llm(json_mode=True), strategy=strategy)

    # nodes ---------------------------------------------------------------

    def _rewrite(self, s: AgentState) -> AgentState:
        tried = s.get("tried", [])
        new = rewrite_queries(self.rewrite_llm, s["question"], avoid=tried)
        new = [q for q in new if q not in tried]
        # First pass: search the user's wording AND the legal rewrites. On a retry the original
        # wording already failed, so re-fusing it would only dilute the new queries.
        queries = new if tried else [s["question"], *new]
        queries = queries or [s["question"]]  # rewriter returned nothing usable
        get_stream_writer()({"type": "rewrite", "queries": new})
        return {
            "queries": queries,
            "trace": [*s.get("trace", []), {"step": "rewrite", "queries": new}],
        }

    def _retrieve(self, s: AgentState) -> AgentState:
        queries = s.get("queries") or [s["question"]]
        chunks = multi_query_search(self.retriever, queries, k=self.k)
        step = {"step": "retrieve", "queries": queries, "found": [c["id"] for c in chunks]}
        get_stream_writer()(
            {"type": "sources", "sources": [_source(c, i) for i, c in enumerate(chunks, 1)]}
        )
        return {
            "chunks": chunks,
            "tried": [*s.get("tried", []), *queries],
            "trace": [*s.get("trace", []), step],
        }

    def _generate(self, s: AgentState) -> AgentState:
        write = get_stream_writer()  # no-op under invoke(); emits events under stream()
        attempts = s.get("attempts", 0) + 1
        write({"type": "generating", "attempt": attempts})
        on_token = (lambda t: write({"type": "token", "text": t})) if self._streaming else None
        ans = answer_question(s["question"], s["chunks"], self.llm, on_token=on_token)
        write({"type": "attempt_done", "attempt": attempts, "supported": ans.supported})
        step = {"step": "generate", "attempt": attempts, "supported": ans.supported}
        return {"answer": ans, "attempts": attempts, "trace": [*s.get("trace", []), step]}

    def _route(self, s: AgentState) -> str:
        if s["answer"].supported or s["attempts"] >= self.max_attempts:
            return END
        return "rewrite"

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("rewrite", self._rewrite)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.add_edge(START, "rewrite" if self.strategy == "rewrite_first" else "retrieve")
        g.add_edge("rewrite", "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_conditional_edges("generate", self._route, {"rewrite": "rewrite", END: END})
        return g.compile()

    # public --------------------------------------------------------------

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        self._streaming = False
        final = self.graph.invoke({"question": question})
        return self._result(final, t0)

    def ask_stream(self, question: str) -> Iterator[dict]:
        """Yield events as the agent works: sources, rewrite, token, attempt_done, final."""
        t0 = time.perf_counter()
        self._streaming = True
        final: dict = {}
        try:
            for mode, chunk in self.graph.stream(
                {"question": question}, stream_mode=["custom", "values"]
            ):
                if mode == "custom":
                    yield chunk
                else:
                    final = chunk
        finally:
            self._streaming = False
        yield {"type": "final", **self._result(final, t0)}

    def _result(self, final: dict, t0: float) -> dict:
        ans: Answer = final["answer"]
        by_id = {c["id"]: c for c in final.get("chunks", [])}
        out = asdict(ans)
        out["sources"] = [
            {
                "label": c.label,
                "id": c.chunk_id,
                "title": c.title,
                "text": _body(by_id[c.chunk_id]) if c.chunk_id in by_id else "",
            }
            for c in ans.citations
        ]
        out["attempts"] = final["attempts"]
        out["strategy"] = self.strategy
        out["trace"] = final["trace"]
        out["timing_ms"] = {"total": round((time.perf_counter() - t0) * 1000)}
        return out


def _body(chunk: dict) -> str:
    return chunk["text"].split("\n", 1)[-1]


def _source(chunk: dict, i: int) -> dict:
    return {
        "tag": f"S{i}",
        "label": cite_label(chunk),
        "title": chunk["title"],
        "id": chunk["id"],
        "text": _body(chunk),
    }
