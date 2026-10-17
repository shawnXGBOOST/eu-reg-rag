"""Hybrid retrieval: dense vectors (Qdrant) + BM25 keywords, merged with Reciprocal Rank Fusion.

Dense search captures meaning ("fines" ~ "administrative penalties");
BM25 captures exact terms ("Article 22", "DPIA", "biometric").
RRF merges the two rankings without needing comparable scores.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from eu_reg_rag.config import settings
from eu_reg_rag.rerank import ScoreFn

RRF_K = 60
# Binding provisions. Recitals explain intent but are not law, so "articles-first" search
# keeps them out of the main results (they can still be fetched separately as context).
BINDING_KINDS = (
    "article",
    "annex",
)  # standard constant from the original RRF paper (Cormack et al., 2009)

STOPWORDS = set(
    "a an and are as at be by for from has have in is it its of on or shall that the "
    "this to was were which will with what how does do can i my we our".split()
)

# A function that maps texts -> vectors. Real one wraps sentence-transformers;
# tests pass a cheap fake so they don't need to download a model.
EncodeFn = Callable[[Sequence[str], bool], list[list[float]]]


@dataclass
class Result:
    chunk: dict
    score: float
    dense_rank: int | None
    bm25_rank: int | None


# ---------- helpers ----------


def load_chunks(path: str | None = None) -> list[dict]:
    path = path or f"{settings.processed_dir}/chunks.jsonl"
    if not Path(path).exists():
        raise SystemExit(f"Missing {path}. Run: python -m eu_reg_rag.ingest")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def search_text(chunk: dict) -> str:
    """Text used for matching: cross-reference-enriched if available, else the original."""
    return chunk.get("index_text") or chunk["text"]


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def get_client() -> QdrantClient:
    if settings.qdrant_mode == "embedded":
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(url=settings.qdrant_url)


def sentence_transformer_encoder(model_name: str | None = None) -> EncodeFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name or settings.embed_model)
    # BGE models expect this prefix on queries (not on documents)
    query_prefix = "Represent this sentence for searching relevant passages: "

    def encode(texts: Sequence[str], is_query: bool = False) -> list[list[float]]:
        if is_query:
            texts = [query_prefix + t for t in texts]
        vecs = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=not is_query)
        return vecs.tolist()

    return encode


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Each ranking is a list of doc ids, best first. Returns id -> fused score."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------- indexing ----------


def build_index(chunks: list[dict], encode: EncodeFn, client: QdrantClient) -> None:
    vectors = encode([search_text(c) for c in chunks], False)
    if client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)
    client.create_collection(
        settings.collection,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=i, vector=v, payload=c)
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
    ]
    for start in range(0, len(points), 256):
        client.upsert(settings.collection, points=points[start : start + 256])


# ---------- retrieval ----------


class HybridRetriever:
    def __init__(
        self,
        chunks: list[dict],
        encode: EncodeFn,
        client: QdrantClient,
        reranker: ScoreFn | None = None,
    ):
        self.chunks = chunks
        self.encode = encode
        self.client = client
        self.reranker = reranker
        self.bm25 = BM25Okapi([tokenize(search_text(c)) for c in chunks])

    def dense_ranking(self, query: str, k: int, kinds: Sequence[str] | None = None) -> list[int]:
        vec = self.encode([query], True)[0]
        flt = None
        if kinds:
            flt = Filter(must=[FieldCondition(key="kind", match=MatchAny(any=list(kinds)))])
        hits = self.client.query_points(
            settings.collection, query=vec, limit=k, query_filter=flt
        ).points
        return [int(h.id) for h in hits]

    def bm25_ranking(self, query: str, k: int, kinds: Sequence[str] | None = None) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked = [i for i in ranked if scores[i] > 0]
        if kinds:
            ranked = [i for i in ranked if self.chunks[i]["kind"] in kinds]
        return ranked[:k]

    def search(
        self,
        query: str,
        k: int | None = None,
        mode: str = "hybrid",
        rerank: bool = True,
        kinds: Sequence[str] | None = None,
        max_per_unit: int | None = None,
    ) -> list[Result]:
        """kinds: restrict results to these chunk kinds, e.g. BINDING_KINDS for articles-first.
        max_per_unit: keep at most N chunks per article/recital/annex, so one long article
        split into many chunks can't crowd every other article out of the top k."""
        k = k or settings.top_k_final
        pool = settings.top_k_retrieve
        dense = self.dense_ranking(query, pool, kinds) if mode in ("dense", "hybrid") else []
        bm25 = self.bm25_ranking(query, pool, kinds) if mode in ("bm25", "hybrid") else []

        fused = rrf_fuse([r for r in (dense, bm25) if r])
        candidates = sorted(fused, key=fused.get, reverse=True)
        if max_per_unit and not (rerank and self.reranker):
            candidates = self._cap_per_unit(candidates, max_per_unit)
        d_rank = {doc: r for r, doc in enumerate(dense, start=1)}
        b_rank = {doc: r for r, doc in enumerate(bm25, start=1)}

        if rerank and self.reranker:
            candidates = candidates[: settings.rerank_pool]
            scores = self.reranker(query, [search_text(self.chunks[i]) for i in candidates])
            ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
            if max_per_unit:
                keep = set(self._cap_per_unit([i for i, _ in ranked], max_per_unit))
                ranked = [(i, sc) for i, sc in ranked if i in keep]
            top = ranked[:k]
        else:
            top = [(i, fused[i]) for i in candidates[:k]]

        return [Result(self.chunks[i], s, d_rank.get(i), b_rank.get(i)) for i, s in top]

    def _cap_per_unit(self, ranked: list[int], n: int) -> list[int]:
        counts: dict[tuple, int] = {}
        out = []
        for i in ranked:
            c = self.chunks[i]
            unit = (c["regulation"], c["kind"], c["number"])
            if counts.get(unit, 0) < n:
                counts[unit] = counts.get(unit, 0) + 1
                out.append(i)
        return out
