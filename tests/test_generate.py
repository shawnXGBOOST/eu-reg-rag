from eu_reg_rag.generate import (
    NO_ANSWER,
    answer_question,
    build_prompt,
    cite_label,
    postprocess,
)


def _c(reg, kind, num, para="", title="T", cid="x"):
    return {
        "id": cid,
        "regulation": reg,
        "kind": kind,
        "number": num,
        "paragraphs": para,
        "title": title,
        "text": f"{reg} header\nbody of {num}",
    }


def test_cite_labels():
    assert cite_label(_c("AI Act", "article", "99", "1 to 3")) == "AI Act Art. 99(1)-(3)"
    assert cite_label(_c("AI Act", "article", "5", "1(f)-(g)")) == "AI Act Art. 5(1)(f)-(g)"
    assert cite_label(_c("GDPR", "article", "9", "2(a)-(f)")) == "GDPR Art. 9(2)(a)-(f)"
    assert cite_label(_c("AI Act", "article", "3", "(1)-(6)")) == "AI Act Art. 3(1)-(6)"
    assert cite_label(_c("GDPR", "article", "22")) == "GDPR Art. 22"
    assert cite_label(_c("AI Act", "article", "99", "part2")) == "AI Act Art. 99"
    assert cite_label(_c("AI Act", "annex", "III", "1 to 3")) == "AI Act Annex III, points 1-3"
    assert cite_label(_c("GDPR", "recital", "71")) == "GDPR Recital 71"


CHUNKS = [
    _c("AI Act", "article", "99", "1 to 3", "Penalties", "aiact-art-99-1"),
    _c("AI Act", "article", "5", "1(a)", "Prohibited AI practices", "aiact-art-5-1"),
]


def test_prompt_has_tagged_sources_without_internal_headers():
    p = build_prompt("max fines?", CHUNKS)
    assert "[S1] AI Act Art. 99(1)-(3) - Penalties" in p
    assert "[S2] AI Act Art. 5(1)(a)" in p
    assert "AI Act header" not in p
    assert p.rstrip().endswith("tags):")


def test_postprocess_maps_tags_and_drops_invalid():
    raw = "<think>hmm</think>Fines reach EUR 35 million [S1]. Some practices are banned [S2][S7]."
    a = postprocess("q", raw, CHUNKS)
    assert a.supported
    assert "[AI Act Art. 99(1)-(3)]" in a.answer
    assert "[AI Act Art. 5(1)(a)]" in a.answer
    assert "S7" not in a.answer and a.invalid_tags == ["[S7]"]
    assert "<think>" not in a.answer
    assert [c.chunk_id for c in a.citations] == ["aiact-art-99-1", "aiact-art-5-1"]


def test_uncited_answer_is_flagged_unsupported():
    a = postprocess("q", "Fines are large.", CHUNKS)
    assert not a.supported and a.citations == []


def test_refusal_detected():
    a = postprocess("q", NO_ANSWER, CHUNKS)
    assert not a.supported and a.answer == NO_ANSWER


def test_answer_question_with_fake_llm_and_empty_context():
    seen = {}

    def fake_llm(system, user):
        seen["user"] = user
        return "Up to EUR 35 million [S1]."

    a = answer_question("max fines?", CHUNKS, fake_llm)
    assert a.answer == "Up to EUR 35 million [AI Act Art. 99(1)-(3)]."
    assert "Question: max fines?" in seen["user"]
    assert answer_question("q", [], fake_llm).supported is False
