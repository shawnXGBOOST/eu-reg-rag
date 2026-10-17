from eu_reg_rag.ingest import chunk_units, html_to_lines, parse_units

# Mimics EUR-Lex layout: list labels in separate table cells, chapter/article headings
SAMPLE = """
<html><body>
<p>REGULATION (EU) 2016/679</p>
<p>Whereas:</p>
<table><tr><td>(1)</td><td>Protection of natural persons is a fundamental right.</td></tr></table>
<table><tr><td>(2)</td><td>The principles should respect fundamental rights.</td></tr></table>
<p>HAVE ADOPTED THIS REGULATION:</p>
<p>CHAPTER I</p><p>General provisions</p>
<p>Article 1</p><p>Subject-matter and objectives</p>
<p>1. This Regulation lays down rules.</p>
<p>2. This Regulation protects rights.</p>
<p>CHAPTER II</p><p>Principles</p>
<p>Article 6</p><p>Lawfulness of processing</p>
<p>1. Processing shall be lawful only if:</p>
<table><tr><td>(a)</td><td>the data subject has given consent;</td></tr></table>
<table><tr><td>(b)</td><td>processing is necessary for a contract;</td></tr></table>
<p>2. Member States may maintain more specific provisions with reference to Article 6(1).</p>
<p>Done at Brussels, 27 April 2016.</p>
<p>ANNEX I</p><p>Union harmonisation legislation</p>
<p>Section A. List based on the New Legislative Framework</p>
<table><tr><td>(1)</td><td>OJ L 119, 4.5.2016, p. 1.</td></tr></table>
</body></html>
"""


def _units():
    return parse_units(html_to_lines(SAMPLE), "GDPR")


def test_counts_and_structure():
    units = _units()
    kinds = [(u.kind, u.number) for u in units]
    assert kinds == [
        ("recital", "1"),
        ("recital", "2"),
        ("article", "1"),
        ("article", "6"),
        ("annex", "I"),
    ]


def test_article_metadata_and_labels_joined():
    art6 = [u for u in _units() if u.kind == "article" and u.number == "6"][0]
    assert art6.title == "Lawfulness of processing"
    assert art6.chapter == "Chapter II: Principles"
    assert "(a) the data subject has given consent;" in art6.lines
    # in-text reference "Article 6(1)" must not start a new article
    assert any("Article 6(1)" in ln for ln in art6.lines)


def test_footnotes_dropped_from_annex():
    annex = [u for u in _units() if u.kind == "annex"][0]
    assert not any("OJ L" in ln for ln in annex.lines)


def _art6(max_words):
    chunks = chunk_units(_units(), max_words=max_words)
    return [c for c in chunks if c.kind == "article" and c.number == "6"]


def test_long_article_split_on_paragraph_boundaries():
    art6 = _art6(max_words=25)
    assert [c.paragraphs for c in art6] == ["1", "2"]
    assert all(c.text.startswith("GDPR Article 6 - Lawfulness of processing") for c in art6)


def test_oversized_paragraph_split_by_points_with_lead_in_repeated():
    art6 = _art6(max_words=12)
    assert [c.paragraphs for c in art6] == ["1(a)", "1(b)", "2"]
    # each point keeps the paragraph's lead-in so it makes sense on its own
    assert all("1. Processing shall be lawful only if:" in c.text for c in art6[:2])
    # the "2." label must stay attached to its sentence
    assert "2. Member States" in art6[2].text


def test_roman_vs_letter_i_resolved_by_lookahead():
    from eu_reg_rag.ingest import _classify_points, _point_range

    # (i) after (h) followed by (ii) -> roman sub-point of (h)
    roman = ["(g) x", "(h) y", "(i) z", "(ii) w"]
    assert _point_range([p for p in _classify_points(roman) if p]) == "(g)-(h)(ii)"
    # (i) after (h) followed by (j) -> the letter i
    letter = ["(h) y", "(i) z", "(j) w"]
    assert [p[0] for p in _classify_points(letter)] == ["letter"] * 3


def test_cross_refs_expanded_but_not_external_ones():
    from eu_reg_rag.ingest import Chunk, Unit, enrich_cross_refs

    units = [
        Unit("AI Act", "article", "5", title="Prohibited AI practices"),
        Unit("AI Act", "article", "28", title="Notifying authorities"),
    ]
    text = (
        "Non-compliance with the AI practices referred to in Article 5 shall be fined. "
        "See also Article 28(2) of Regulation (EU) 2019/1020 and Article 5 again."
    )
    c = Chunk("x", "AI Act", "article", "99", "Penalties", "", "", text)
    enrich_cross_refs([c], units)
    assert "Article 5 [Prohibited AI practices] shall" in c.index_text
    assert "Article 28(2) of Regulation" in c.index_text  # external act: untouched
    assert c.index_text.count("[Prohibited AI practices]") == 1  # expanded once only
    assert c.text == text  # original preserved for display
