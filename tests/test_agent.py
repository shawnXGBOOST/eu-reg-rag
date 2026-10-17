import pytest
from qdrant_client import QdrantClient

from eu_reg_rag.agent import RAGAgent, multi_query_search, parse_queries
from eu_reg_rag.generate import NO_ANSWER
from eu_reg_rag.retrieval import HybridRetriever, build_index
from tests.test_retrieval import fake_encode


def _chunk(cid, reg, num, title, text):
    return {
        "id": cid,
        "regulation": reg,
        "kind": "article",
        "number": num,
        "paragraphs": "",
        "title": title,
        "text": text,
    }


CHUNKS = [
    _chunk(
        "gdpr-art-82",
        "GDPR",
        "82",
        "Right to compensation and liability",
        "GDPR Article 82 right to compensation material or non-material damage controller",
    ),
    _chunk(
        "gdpr-art-13",
        "GDPR",
        "13",
        "Information to be provided",
        "GDPR Article 13 information provided company personal data misused collected",
    ),
    _chunk(
        "aiact-art-26",
        "AI Act",
        "26",
        "Obligations of deployers",
        "AI Act Article 26 obligations of deployers of high-risk AI systems",
    ),
]


@pytest.fixture(scope="module")
def retriever():
    client = QdrantClient(":memory:")
    build_index(CHUNKS, fake_encode, client)
    return HybridRetriever(CHUNKS, fake_encode, client)


def answer_if_compensation(system, user):
    """Fake answer LLM: only answers when an Art. 82 source is in the prompt."""
    for line in user.splitlines():
        if line.startswith("[S") and "Art. 82" in line:
            return f"You can claim compensation {line.split()[0]}."
    return NO_ANSWER


def legal_rewriter(system, user):
    return '{"queries": ["right to compensation damage"]}'


def test_parse_queries_handles_json_and_junk():
    assert parse_queries('{"queries": ["a b c", "A B C", "d e f"]}') == ["a b c", "d e f"]
    assert parse_queries('Sure! {"queries": ["deployer obligations"]') == ["deployer obligations"]
    assert parse_queries("no json here") == []
    assert len(parse_queries('{"queries": ["q1x", "q2x", "q3x", "q4x"]}')) == 3


def test_multi_query_search_fuses_and_keeps_one_per_article(retriever):
    out = multi_query_search(retriever, ["company misused data", "right to compensation"], k=3)
    ids = [c["id"] for c in out]
    assert "gdpr-art-82" in ids and len(ids) == len(set(ids))


def test_fallback_rewrites_only_after_refusal(retriever):
    agent = RAGAgent(retriever, answer_if_compensation, legal_rewriter, strategy="fallback", k=1)
    out = agent.ask("Can I sue a company for money if they misused my data?")
    steps = [t["step"] for t in out["trace"]]
    assert steps == ["retrieve", "generate", "rewrite", "retrieve", "generate"]
    assert out["supported"] and out["attempts"] == 2
    assert out["sources"][0]["id"] == "gdpr-art-82"


def test_fallback_skips_rewrite_when_first_answer_is_supported(retriever):
    agent = RAGAgent(retriever, answer_if_compensation, legal_rewriter, strategy="fallback", k=1)
    out = agent.ask("right to compensation")
    assert [t["step"] for t in out["trace"]] == ["retrieve", "generate"]


def test_rewrite_first_and_attempt_limit(retriever):
    def never(system, user):
        return NO_ANSWER

    agent = RAGAgent(retriever, never, legal_rewriter, strategy="rewrite_first", max_attempts=2)
    out = agent.ask("anything")
    steps = [t["step"] for t in out["trace"]]
    assert steps[0] == "rewrite" and steps.count("generate") == 2
    assert not out["supported"]
    # the original question is always searched alongside rewrites
    assert out["trace"][1]["queries"][0] == "anything"


def test_invalid_strategy_rejected(retriever):
    with pytest.raises(ValueError):
        RAGAgent(retriever, never := (lambda s, u: ""), strategy="yolo")  # noqa: F841


def test_ask_stream_emits_events_in_order(retriever):
    def streaming_llm(system, user, on_token=None):
        text = "Compensation is available [S1]."
        if on_token:
            for word in text.split(" "):
                on_token(word + " ")
        return text

    agent = RAGAgent(retriever, streaming_llm, legal_rewriter, strategy="fallback", k=1)
    events = list(agent.ask_stream("right to compensation"))
    types = [e["type"] for e in events]
    assert types[0] == "sources" and types[-1] == "final"
    assert types.count("token") == 4  # one per word, streamed live
    assert events[0]["sources"][0]["label"] == "GDPR Art. 82"
    final = events[-1]
    assert final["supported"] and final["sources"][0]["text"]  # cited article text included
    # invoke() path must still work without streaming
    assert agent.ask("right to compensation")["supported"]
