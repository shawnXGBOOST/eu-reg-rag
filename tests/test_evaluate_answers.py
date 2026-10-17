import json

from eu_reg_rag.evaluate_answers import chunk_article_key, score, to_markdown


def test_chunk_article_key():
    assert chunk_article_key("aiact-art-99-1") == "aiact-art-99"
    assert chunk_article_key("gdpr-art-22") == "gdpr-art-22"
    assert chunk_article_key("aiact-ann-III-2") == "aiact-ann-III"


def _row(supported, gold_cited=False, attempts=1, invalid=(), qid="q", kind="easy"):
    return {
        "id": qid,
        "question": "?",
        "kind": kind,
        "gold": ["g"],
        "supported": supported,
        "cited": ["g"] if gold_cited else ["x"],
        "gold_cited": gold_cited,
        "invalid_tags": list(invalid),
        "attempts": attempts,
        "seconds": 10.0,
    }


def test_score_and_markdown():
    pos = [
        _row(True, True),
        _row(True, False, attempts=2),
        _row(False),
        _row(True, True, invalid=["[S9]"]),
    ]
    neg = [_row(False, kind="off-topic"), _row(True, kind="near-miss", qid="n2")]
    s = score(pos, neg)
    assert s["answer_rate"] == 0.75
    assert abs(s["citation_accuracy"] - 2 / 3) < 1e-9
    assert s["invalid_citation_rate"] == 0.25
    assert s["rescued_by_retry"] == 1
    assert s["refusal_accuracy"] == 0.5
    md = to_markdown(s, pos, neg, "fallback")
    assert "answered an out-of-scope question" in md and "refused:" in md


def test_negatives_file_is_valid():
    with open("eval/negatives.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) >= 10 and all("gold" not in r for r in rows)
