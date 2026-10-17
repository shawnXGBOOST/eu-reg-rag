"""Parse EUR-Lex regulation HTML into structure-aware chunks.

Pipeline: HTML -> clean text lines -> structured units (recitals, articles, annexes)
-> chunks (one per unit; long articles split on paragraph boundaries).

Parsing works on text lines rather than CSS classes, because EUR-Lex has used
different HTML class schemes over the years (GDPR 2016 vs AI Act 2024).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

RE_ARTICLE = re.compile(r"^Article (\d+[a-z]?)$")
RE_CHAPTER = re.compile(r"^CHAPTER ([IVXLC]+)$")
RE_SECTION = re.compile(r"^SECTION \d+$")
RE_ANNEX = re.compile(r"^ANNEX ([IVXLC]+)$")
RE_RECITAL = re.compile(r"^\((\d+)\)\s+(.+)$")
RE_FOOTNOTE = re.compile(r"^\(\d+\)\s+OJ [A-Z]")
RE_LABEL_ONLY = re.compile(r"^(\(\w{1,5}\)|\d{1,3}\.)$")  # "(a)", "(iv)", "3." alone
RE_PARAGRAPH = re.compile(r"^(\d{1,3})\.\s+")

RECITALS_START = "Whereas:"
RECITALS_END = "HAVE ADOPTED THIS REGULATION"
ARTICLES_END = "Done at "


@dataclass
class Unit:
    regulation: str  # "GDPR" | "AI Act"
    kind: str  # "recital" | "article" | "annex"
    number: str
    title: str = ""
    chapter: str = ""
    lines: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    regulation: str
    kind: str
    number: str
    title: str
    chapter: str
    paragraphs: str  # e.g. "1-3" for split articles, "" if whole unit
    text: str  # original text: shown to users and the LLM
    index_text: str = ""  # text used for search: cross-references expanded with titles


# ---------- HTML -> lines ----------


def html_to_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    raw = [re.sub(r"\s+", " ", ln).strip() for ln in soup.get_text("\n").split("\n")]
    raw = [ln for ln in raw if ln]

    # EUR-Lex puts list labels in their own table cell: "(a)" / "text". Re-join them.
    lines: list[str] = []
    i = 0
    while i < len(raw):
        if RE_LABEL_ONLY.match(raw[i]) and i + 1 < len(raw):
            lines.append(f"{raw[i]} {raw[i + 1]}")
            i += 2
        else:
            lines.append(raw[i])
            i += 1
    return lines


# ---------- lines -> units ----------


def _clean_title(title: str) -> str:
    # the official AI Act text has typos like "Subject matter`"
    return title.strip().strip("`'‘’").strip()


def parse_units(lines: list[str], regulation: str) -> list[Unit]:
    units: list[Unit] = []
    current: Unit | None = None
    chapter = ""
    state = "preamble"  # preamble -> recitals -> articles -> tail -> annexes
    i = 0

    def flush():
        nonlocal current
        if current and current.lines:
            units.append(current)
        current = None

    while i < len(lines):
        ln = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""

        if state == "preamble":
            if ln.startswith(RECITALS_START):
                state = "recitals"
        elif state == "recitals":
            if ln.startswith(RECITALS_END):
                flush()
                state = "articles"
            elif m := RE_RECITAL.match(ln):
                flush()
                current = Unit(regulation, "recital", m.group(1), lines=[m.group(2)])
            elif current:
                current.lines.append(ln)
        elif state in ("articles", "tail", "annexes"):
            if m := RE_ANNEX.match(ln):
                flush()
                current = Unit(regulation, "annex", m.group(1), title=_clean_title(nxt))
                state = "annexes"
                i += 2
                continue
            if state == "articles":
                if m := RE_CHAPTER.match(ln):
                    flush()
                    chapter = f"Chapter {m.group(1)}: {nxt}"
                    i += 2
                    continue
                if RE_SECTION.match(ln):
                    i += 2  # skip "SECTION n" + its title
                    continue
                if m := RE_ARTICLE.match(ln):
                    flush()
                    current = Unit(
                        regulation, "article", m.group(1), title=_clean_title(nxt), chapter=chapter
                    )
                    i += 2
                    continue
                if ln.startswith(ARTICLES_END):
                    flush()
                    state = "tail"
                elif current:
                    current.lines.append(ln)
            elif state == "annexes" and current and not RE_FOOTNOTE.match(ln):
                current.lines.append(ln)
        i += 1

    flush()
    return units


# ---------- units -> chunks ----------


def _header(u: Unit) -> str:
    label = {"recital": "Recital", "article": "Article", "annex": "Annex"}[u.kind]
    head = f"{u.regulation} {label} {u.number}"
    return f"{head} - {u.title}" if u.title else head


def _split_paragraphs(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Group lines under their numbered paragraph ("1.", "2." ...)."""
    groups: list[tuple[str, list[str]]] = []
    for ln in lines:
        m = RE_PARAGRAPH.match(ln)
        if m or not groups:
            groups.append((m.group(1) if m else "", [ln]))
        else:
            groups[-1][1].append(ln)
    return groups


def _make_chunk(u: Unit, chunk_id: str, paragraphs: str, body: str) -> Chunk:
    return Chunk(
        chunk_id,
        u.regulation,
        u.kind,
        u.number,
        u.title,
        u.chapter,
        paragraphs,
        f"{_header(u)}\n{body}",
    )


RE_POINT = re.compile(r"^\((\w{1,5})\)\s")
RE_SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")
LEAD_MAX_WORDS = 80  # lead-in lines longer than this are not repeated in each sub-chunk


def _wc(text: str) -> int:
    return len(text.split())


def _pack(items: list[str], budget: int) -> list[list[str]]:
    """Greedily pack text items into groups whose word count stays <= budget."""
    groups: list[list[str]] = [[]]
    for item in items:
        if groups[-1] and _wc(" ".join(groups[-1] + [item])) > budget:
            groups.append([])
        groups[-1].append(item)
    return groups


def _explode(lines: list[str], budget: int) -> list[str]:
    """Break any single line longer than budget into sentence-sized pieces."""
    out: list[str] = []
    for ln in lines:
        if _wc(ln) <= budget:
            out.append(ln)
        else:
            pieces: list[str] = []
            for s in RE_SENTENCE_END.split(ln):
                # "2." / "(a)" alone is a label, not a sentence: glue it to what follows
                if pieces and RE_LABEL_ONLY.match(pieces[-1]):
                    pieces[-1] = f"{pieces[-1]} {s}"
                else:
                    pieces.append(s)
            out += [" ".join(g) for g in _pack(pieces, budget)]
    return out


# Single letters that are also roman numerals, mapped to the letter that precedes them
ROMAN_LOOKALIKE = {"i": "h", "v": "u", "x": "w"}
ROMAN_SUCCESSOR = {"i": "ii", "v": "vi", "x": "xi"}

Point = tuple[str, str, str]  # (kind, token, parent letter)


def _classify_points(items: list[str]) -> list[Point | None]:
    """Classify each item's leading point label as num / letter / roman, using lookahead
    to resolve "(i)": after "(h)" it's the letter i, unless "(ii)" follows."""
    toks = [m.group(1) if (m := RE_POINT.match(it)) else None for it in items]
    out: list[Point | None] = []
    prev_letter = ""
    for idx, tok in enumerate(toks):
        if tok is None:
            out.append(None)
            continue
        if tok.isdigit():
            kind = "num"
        elif re.fullmatch(r"[ivxl]+", tok):
            kind = "roman"
            if len(tok) == 1 and ROMAN_LOOKALIKE.get(tok) == prev_letter:
                nxt = next((t for t in toks[idx + 1 :] if t), None)
                kind = "roman" if nxt == ROMAN_SUCCESSOR[tok] else "letter"
        else:
            kind = "letter"
        out.append((kind, tok, prev_letter))
        if kind == "letter":
            prev_letter = tok
    return out


def _point_range(points: list[Point]) -> str:
    """Label like "(a)-(f)", "(c)(ii)-(f)" or "(g)-(h)(iii)". Ranges are expressed in
    the top-level point kind so sub-points don't produce nonsense like "(ii)-(f)"."""
    if not points:
        return ""

    def fmt(p: Point) -> str:
        kind, tok, parent = p
        return f"({parent})({tok})" if kind == "roman" and parent else f"({tok})"

    first, last = points[0], points[-1]
    # "(c)" ... "(c)(ii)" is all inside point (c): label it just "(c)"
    if len(points) == 1 or (first[0] == "letter" and last[0] == "roman" and last[2] == first[1]):
        return fmt(first)
    return f"{fmt(first)}-{fmt(last)}"


POINT_MAX_WORDS = 150  # lists of points are split finer: each point is a distinct rule


def _segments(u: Unit, max_words: int) -> list[tuple[str, str, bool]]:
    """(label, text, standalone) segments. Paragraphs that fit stay whole. Oversized ones are
    split by points (or sentences), repeating the lead-in line for context; those pieces are
    standalone so they never get re-merged with neighbours."""
    segs: list[tuple[str, str, bool]] = []
    for num, para_lines in _split_paragraphs(u.lines):
        text = "\n".join(para_lines)
        if _wc(text) <= max_words:
            segs.append((num, text, False))
            continue
        lead, rest = para_lines[0], para_lines[1:]
        if _wc(lead) > LEAD_MAX_WORDS or not rest:
            lead, rest = "", para_lines
        has_points = any(RE_POINT.match(ln) for ln in rest)
        budget = (min(POINT_MAX_WORDS, max_words) if has_points else max_words) - _wc(lead)
        items = _explode(rest, budget)
        kinds = _classify_points(items)
        pos = 0
        for group in _pack(items, budget):
            pts = [p for p in kinds[pos : pos + len(group)] if p]
            pos += len(group)
            label = f"{num}{_point_range(pts)}" or num
            segs.append((label, "\n".join([lead, *group] if lead else group), True))
    return segs


def chunk_units(units: list[Unit], max_words: int = 300) -> list[Chunk]:
    chunks: list[Chunk] = []
    for u in units:
        slug = "gdpr" if u.regulation == "GDPR" else "aiact"
        base_id = f"{slug}-{u.kind[:3]}-{u.number}"
        body = "\n".join(u.lines)

        if _wc(body) <= max_words:
            chunks.append(_make_chunk(u, base_id, "", body))
            continue

        # Pack segments (whole paragraphs where possible) into chunks of <= max_words.
        groups: list[list[tuple[str, str]]] = [[]]
        for label, text, standalone in _segments(u, max_words):
            current = " ".join(t for _, t in groups[-1])
            if groups[-1] and (standalone or _wc(f"{current} {text}") > max_words):
                groups.append([])
            groups[-1].append((label, text))
            if standalone:
                groups.append([])
        groups = [g for g in groups if g]

        for part, group in enumerate(groups, start=1):
            labels = [lbl for lbl, _ in group if lbl]
            if not labels:
                label = f"part{part}"
            elif len(labels) == 1:
                label = labels[0]
            else:
                label = f"{labels[0]} to {labels[-1]}"
            text = "\n".join(t for _, t in group)
            chunks.append(_make_chunk(u, f"{base_id}-{part}", label, text))
    return chunks


# ---------- cross-reference enrichment ----------

# "Article 5", "Article 5(1)(h)". \b stops "Article 28" being read as "Article 2" + "8".
RE_XREF = re.compile(r"\bArticle (\d+)\b((?:\(\w+\))*)")
# A reference into ANOTHER act: "Article 4, point (14), of Regulation (EU) 2016/679"
RE_EXTERNAL = re.compile(
    r"^[^.;]{0,60}?\b(?:of|to) (?:Regulation|Directive|Decision|the Treaty|TFEU)"
)


def enrich_cross_refs(chunks: list[Chunk], units: list[Unit]) -> None:
    """Fill index_text: legal text says "the practices referred to in Article 5" rather than
    "prohibited AI practices", which defeats both keyword and semantic search. Appending the
    referenced article's title ("Article 5 [Prohibited AI practices]") restores the link."""
    titles = {u.number: u.title for u in units if u.kind == "article"}
    for c in chunks:
        seen: set[str] = {c.number} if c.kind == "article" else set()

        def expand(m: re.Match, seen: set[str] = seen, text: str = c.text) -> str:
            num = m.group(1)
            tail = text[m.end() :].split("Article", 1)[0]  # stop at the next reference
            if num in seen or num not in titles or RE_EXTERNAL.match(tail):
                return m.group(0)
            seen.add(num)
            return f"{m.group(0)} [{titles[num]}]"

        c.index_text = RE_XREF.sub(expand, c.text)


# ---------- CLI ----------

DOCS = {"gdpr.html": "GDPR", "ai_act.html": "AI Act"}
EXPECTED = {
    "GDPR": {"article": 99, "recital": 173, "annex": 0},
    "AI Act": {"article": 113, "recital": 180, "annex": 13},
}


def main(raw_dir: str = "data/raw", out_path: str = "data/processed/chunks.jsonl") -> None:
    all_chunks: list[Chunk] = []
    for fname, reg in DOCS.items():
        path = Path(raw_dir) / fname
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run: python scripts/download.py")
        units = parse_units(html_to_lines(path.read_text(encoding="utf-8")), reg)
        counts = {k: sum(u.kind == k for u in units) for k in ("article", "recital", "annex")}
        ok = "OK" if counts == EXPECTED[reg] else f"MISMATCH (expected {EXPECTED[reg]})"
        print(f"{reg}: {counts} -> {ok}")
        chunks = chunk_units(units)
        enrich_cross_refs(chunks, units)
        all_chunks += chunks

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    words = [len(c.text.split()) for c in all_chunks]
    print(
        f"Wrote {len(all_chunks)} chunks to {out_path} "
        f"(avg {sum(words) // len(words)} words, max {max(words)})"
    )


if __name__ == "__main__":
    main()
