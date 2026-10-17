"""Retrieval tests with a fake encoder and in-memory Qdrant (no model download)."""

import hashlib

import pytest
from qdrant_client import QdrantClient

from eu_reg_rag.retrieval import HybridRetriever, build_index, rrf_fuse, tokenize

DIM = 64


def fake_encode(texts, is_query=False):
    """Hashed bag-of-words vectors: texts sharing words get similar vectors."""
    out = []
    for t in texts:
        v = [0.0] * DIM
        for tok in tokenize(t):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


CHUNKS = [
    {"id": "gdpr-art-35", "text": "GDPR Article 35 data protection impact assessment high risk"},
    {"id": "gdpr-art-83", "text": "GDPR Article 83 administrative fines penalties infringements"},
    {"id": "aiact-art-5", "text": "AI Act Article 5 prohibited practices biometric categorisation"},
    {"id": "aiact-art-6", "text": "AI Act Article 6 classification high-risk AI systems"},
]


@pytest.fixture(scope="module")
def retriever():
    client = QdrantClient(":memory:")
    build_index(CHUNKS, fake_encode, client)
    return HybridRetriever(CHUNKS, fake_encode, client)


def test_rrf_rewards_agreement():
    # doc 2 is ranked 2nd by both -> beats docs ranked 1st by only one list
    scores = rrf_fuse([[1, 2, 3], [4, 2, 5]])
    assert max(scores, key=scores.get) == 2


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_modes_find_exact_match(retriever, mode):
    top = retriever.search("biometric categorisation prohibited", k=1, mode=mode)[0]
    assert top.chunk["id"] == "aiact-art-5"


def test_hybrid_reports_both_ranks(retriever):
    r = retriever.search("administrative fines", k=1)[0]
    assert r.chunk["id"] == "gdpr-art-83"
    assert r.dense_rank == 1 and r.bm25_rank == 1


def test_bm25_ignores_zero_score_docs(retriever):
    assert retriever.bm25_ranking("xyzzy nonsense", k=10) == []


def test_reranker_reorders_candidates():
    client = QdrantClient(":memory:")
    build_index(CHUNKS, fake_encode, client)

    def prefer_art6(query, texts):  # fake cross-encoder: loves Article 6
        return [10.0 if "Article 6" in t else 0.0 for t in texts]

    r = HybridRetriever(CHUNKS, fake_encode, client, reranker=prefer_art6)
    query = "high risk"  # matches both Art. 35 and Art. 6
    assert r.search(query, k=1)[0].chunk["id"] == "aiact-art-6"
    # rerank=False falls back to the fused order, still returning valid results
    assert len(r.search(query, k=2, rerank=False)) == 2
