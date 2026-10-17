"""Grounded answer generation with verifiable citations.

The LLM sees retrieved chunks tagged [S1]..[Sn] and must cite those tags. We then:
  - map each tag to a real legal citation ("AI Act Art. 99(3)"),
  - drop tags that don't exist (hallucinated citations are caught, not trusted),
  - flag answers with no valid citation as unsupported.
Citing short tags instead of free-text article numbers is far more reliable for small models.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from eu_reg_rag.config import settings

# llm(system, user) -> completion text. Backends may also accept on_token=callback to stream.
LLMFn = Callable[..., str]
TokenFn = Callable[[str], None]

NO_ANSWER = "The provided articles do not answer this question."

SYSTEM_PROMPT = f"""You are a precise assistant for the EU AI Act and the GDPR.
Answer ONLY from the numbered sources provided. Rules:
1. End every factual sentence with the tag(s) of its supporting source(s): [S1] or [S2][S3].
2. Use only tags that appear in the sources. Never invent article numbers.
3. If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
4. Be concise: direct answer first, then key conditions or exceptions. No disclaimers.
"""

RE_TAG = re.compile(r"\[S(\d+)\]")
RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


# ---------- citation labels ----------


def _fmt_part(x: str) -> str:
    """'1' -> '(1)', '2(a)' -> '(2)(a)', '(33)' -> '(33)'."""
    m = re.match(r"^(\d+)(.*)$", x)
    return f"({m.group(1)}){m.group(2)}" if m else x


def cite_label(chunk: dict) -> str:
    """Human legal citation for a chunk, e.g. 'AI Act Art. 5(1)(f)-(g)', 'GDPR Art. 6(1)-(2)'."""
    reg, kind, num, para = chunk["regulation"], chunk["kind"], chunk["number"], chunk["paragraphs"]
    if kind == "recital":
        return f"{reg} Recital {num}"
    if kind == "annex":
        base = f"{reg} Annex {num}"
        if not para or para.startswith("part"):
            return base
        return f"{base}, points {para.replace(' to ', '-')}"
    base = f"{reg} Art. {num}"
    if not para or para.startswith("part"):
        return base
    if " to " in para:
        a, b = para.split(" to ", 1)
        return f"{base}{_fmt_part(a)}-{_fmt_part(b)}"
    return f"{base}{_fmt_part(para)}"


# ---------- prompt ----------


def build_prompt(question: str, chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        body = c["text"].split("\n", 1)[-1]  # drop our own header line; label replaces it
        parts.append(f"[S{i}] {cite_label(c)} - {c['title']}\n{body}")
    sources = "\n\n".join(parts)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}\n\nAnswer (cite sources with [S#] tags):"


# ---------- post-processing ----------


@dataclass
class Citation:
    tag: str
    label: str
    chunk_id: str
    title: str


@dataclass
class Answer:
    question: str
    answer: str  # with [S#] tags replaced by legal citations
    raw: str  # model output as returned
    citations: list[Citation] = field(default_factory=list)
    invalid_tags: list[str] = field(default_factory=list)
    supported: bool = True


def postprocess(question: str, raw: str, chunks: list[dict]) -> Answer:
    text = RE_THINK.sub("", raw).strip()
    if NO_ANSWER.lower().rstrip(".") in text.lower():
        return Answer(question, NO_ANSWER, raw, supported=False)

    used: dict[int, Citation] = {}
    invalid: list[str] = []

    def replace(m: re.Match) -> str:
        n = int(m.group(1))
        if not 1 <= n <= len(chunks):
            invalid.append(m.group(0))
            return ""
        c = chunks[n - 1]
        used.setdefault(n, Citation(f"S{n}", cite_label(c), c["id"], c["title"]))
        return f"[{used[n].label}]"

    answer = RE_TAG.sub(replace, text)
    answer = re.sub(r"\]\s*\[", "] [", answer)  # space out adjacent citations
    answer = re.sub(r"[ \t]+([.,;])", r"\1", answer).strip()
    return Answer(
        question,
        answer,
        raw,
        citations=[used[n] for n in sorted(used)],
        invalid_tags=invalid,
        supported=bool(used),
    )


# ---------- LLM backend ----------


def ollama_llm(model: str | None = None, timeout: float = 180.0, json_mode: bool = False) -> LLMFn:
    def call(system: str, user: str, on_token: TokenFn | None = None) -> str:
        payload = {
            "model": model or settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": on_token is not None,
            "think": False,  # Qwen3: skip the hidden reasoning pass; much faster on CPU
            "options": {"temperature": settings.llm_temperature, "num_ctx": 8192},
        }
        if json_mode:
            payload["format"] = "json"  # Ollama constrains output to valid JSON
        url = f"{settings.ollama_url}/api/chat"
        if on_token is None:
            r = httpx.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]

        # Streaming: Ollama sends one JSON object per line, each with a piece of the reply.
        parts: list[str] = []
        with httpx.stream("POST", url, json=payload, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                piece = json.loads(line).get("message", {}).get("content", "")
                if piece:
                    parts.append(piece)
                    on_token(piece)
        return "".join(parts)

    return call


class _ThinkFilter:
    """Streams tokens to the UI but hides a leading <think>...</think> block (Qwen3 on hosted
    APIs may emit one even when asked not to). postprocess() strips it from the final text too."""

    def __init__(self, on_token: TokenFn):
        self.on_token, self.buf, self.state = on_token, "", "start"

    def feed(self, piece: str) -> None:
        if self.state == "pass":
            self.on_token(piece)
            return
        self.buf += piece
        text = self.buf.lstrip()
        if self.state == "start":
            if text.startswith("<think>"):
                self.state = "thinking"
            elif "<think>".startswith(text):
                return  # not enough text yet to know
            else:
                self.state, out, self.buf = "pass", self.buf, ""
                self.on_token(out)
                return
        if self.state == "thinking" and "</think>" in self.buf:
            rest = self.buf.split("</think>", 1)[1].lstrip()
            self.state, self.buf = "pass", ""
            if rest:
                self.on_token(rest)

    def flush(self) -> None:
        if self.state == "start" and self.buf:
            self.on_token(self.buf)


def openai_llm(model: str | None = None, timeout: float = 120.0, json_mode: bool = False) -> LLMFn:
    """Any OpenAI-compatible /chat/completions endpoint (HF router, Groq, OpenRouter ...)."""

    def call(system: str, user: str, on_token: TokenFn | None = None) -> str:
        key = settings.llm_api_key or settings.hf_token
        if not key:
            raise httpx.HTTPError("LLM_API_KEY / HF_TOKEN is not set")
        payload = {
            "model": model or settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                # "/no_think" is Qwen3's switch for skipping the reasoning pass
                {"role": "user", "content": f"{user}\n/no_think"},
            ],
            "temperature": settings.llm_temperature,
            "stream": on_token is not None,
        }
        if json_mode:
            payload["messages"][0]["content"] += "\nRespond with valid JSON only."
        if settings.llm_extra_body:
            payload.update(json.loads(settings.llm_extra_body))
        url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
        if on_token is None:
            r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""

        parts: list[str] = []
        filt = _ThinkFilter(on_token)
        with httpx.stream("POST", url, json=payload, headers=headers, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                choices = json.loads(line[5:]).get("choices") or [{}]
                piece = choices[0].get("delta", {}).get("content") or ""
                if piece:
                    parts.append(piece)
                    filt.feed(piece)
        filt.flush()
        return "".join(parts)

    return call


def make_llm(json_mode: bool = False) -> LLMFn:
    """The LLM chosen by settings.llm_backend ("ollama" or "openai")."""
    if settings.llm_backend == "openai":
        return openai_llm(json_mode=json_mode)
    return ollama_llm(json_mode=json_mode)


def answer_question(
    question: str, chunks: list[dict], llm: LLMFn, on_token: TokenFn | None = None
) -> Answer:
    if not chunks:
        return Answer(question, NO_ANSWER, "", supported=False)
    prompt = build_prompt(question, chunks)
    raw = llm(SYSTEM_PROMPT, prompt, on_token=on_token) if on_token else llm(SYSTEM_PROMPT, prompt)
    return postprocess(question, raw, chunks)
