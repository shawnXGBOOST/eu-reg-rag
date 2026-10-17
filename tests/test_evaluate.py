import json

from eu_reg_rag.evaluate import QResult, article_key, first_hit_rank, summarize, to_markdown


def _chunk(reg, kind, num):
    return {"regulation": reg, "kind": kind, "number": num}


def test_article_key_ignores_paragraph_split():
    assert article_key(_chunk("AI Act", "article", "99")) == "aiact-art-99"
    assert article_key(_chunk("GDPR", "recital", "71")) == "gdpr-rec-71"
    assert article_key(_chunk("AI Act", "annex", "III")) == "aiact-ann-III"


def test_first_hit_rank():
    retrieved = [_chunk("AI Act", "article", "5"), _chunk("AI Act", "article", "99")]
    assert first_hit_rank(retrieved, ["aiact-art-99"]) == 2
    assert first_hit_rank(retrieved, ["gdpr-art-6"]) is None


def _qr(rank, difficulty="easy"):
    return QResult("q", difficulty, "?", ["x"], rank, [], 10.0)


def test_summarize_metrics():
    s = summarize([_qr(1), _qr(3), _qr(None), _qr(7)])
    assert s["hit@1"] == 0.25
    assert s["hit@5"] == 0.5
    assert abs(s["mrr@10"] - (1 + 1 / 3 + 1 / 7) / 4) < 1e-9


def test_markdown_lists_misses():
    md = to_markdown({"A": [_qr(1), _qr(None, "hard")]})
    assert "| A | 0.50 | 0.50 |" in md
    assert "Misses" in md and "(hard)" in md


def test_question_file_is_valid():
    with open("eval/questions.jsonl", encoding="utf-8") as f:
        qs = [json.loads(line) for line in f]
    assert len(qs) >= 40
    assert len({q["id"] for q in qs}) == len(qs)
    assert all(q["difficulty"] in {"easy", "paraphrase", "hard"} and q["gold"] for q in qs)
