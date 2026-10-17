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


def test_articles_first_excludes_recitals():
    chunks = CHUNKS + [
        {"id": "gdpr-rec-91", "kind": "recital", "text": "Recital impact assessment high risk"}
    ]
    chunks = [{**c, "kind": c.get("kind", "article")} for c in chunks]
    client = QdrantClient(":memory:")
    build_index(chunks, fake_encode, client)
    r = HybridRetriever(chunks, fake_encode, client)
    q = "impact assessment high risk"
    assert "gdpr-rec-91" in [x.chunk["id"] for x in r.search(q, k=5, rerank=False)]
    filtered = r.search(q, k=5, rerank=False, kinds=("article", "annex"))
    assert filtered and all(x.chunk["kind"] == "article" for x in filtered)


def test_max_per_unit_caps_chunks_of_same_article():
    chunks = [
        {
            "id": "a-1",
            "regulation": "GDPR",
            "kind": "article",
            "number": "6",
            "text": "lawful basis consent",
        },
        {
            "id": "a-2",
            "regulation": "GDPR",
            "kind": "article",
            "number": "6",
            "text": "lawful basis contract",
        },
        {
            "id": "b",
            "regulation": "GDPR",
            "kind": "article",
            "number": "7",
            "text": "consent conditions lawful",
        },
    ]
    client = QdrantClient(":memory:")
    build_index(chunks, fake_encode, client)
    r = HybridRetriever(chunks, fake_encode, client)
    ids = [
        x.chunk["id"] for x in r.search("lawful basis consent", k=3, rerank=False, max_per_unit=1)
    ]
    assert len(ids) == 2 and "b" in ids  # only one of a-1/a-2 survives


def _defn(cid, num, term, text):
    return {
        "id": cid,
        "regulation": "GDPR",
        "kind": "article",
        "number": "4",
        "paragraphs": f"({num})",
        "title": "Definitions",
        "text": f"GDPR Article 4 - Definitions: definition of {term}\n({num}) ‘{term}’ {text}",
    }


def test_definitional_question_promotes_the_exact_defined_term():
    chunks = [
        _defn("d1", 1, "personal data", "means any information relating to a person"),
        _defn("d2", 12, "personal data breach", "means a breach of security of personal data"),
        _defn("d3", 10, "third party", "means a body other than the data subject, personal data"),
    ]
    client = QdrantClient(":memory:")
    build_index(chunks, fake_encode, client)
    r = HybridRetriever(chunks, fake_encode, client)
    top = r.search("What is the definition of personal data?", k=3, rerank=False)
    assert top[0].chunk["id"] == "d1"
    # not a definitional question: no promotion
    assert r._defined_terms_in("Can I process personal data without consent?", None) == []
    # longest match wins when both "personal data" and "personal data breach" are named
    assert r._defined_terms_in("what is a personal data breach", None) == [1]
