import json

import httpx
import pytest
from fastapi import HTTPException

from eu_reg_rag import api, generate
from eu_reg_rag.config import settings
from eu_reg_rag.generate import _ThinkFilter, make_llm, openai_llm


def _collect(pieces):
    out = []
    f = _ThinkFilter(out.append)
    for p in pieces:
        f.feed(p)
    f.flush()
    return "".join(out)


def test_think_filter_hides_leading_think_block_only():
    assert _collect(["<thi", "nk>\n\n</thi", "nk>\n\nHello ", "world"]) == "Hello world"
    assert _collect(["Hello ", "<think>", "x"]) == "Hello <think>x"  # not leading: untouched
    assert _collect(["<th"]) == "<th"  # short reply that merely starts like a tag is not lost


def test_make_llm_follows_backend_setting(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    assert make_llm().__qualname__.startswith("ollama_llm")
    monkeypatch.setattr(settings, "llm_backend", "openai")
    assert make_llm().__qualname__.startswith("openai_llm")


def test_openai_llm_requires_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "hf_token", "")
    with pytest.raises(httpx.HTTPError):
        openai_llm()("sys", "hi")


def test_openai_llm_request_and_response(monkeypatch):
    monkeypatch.setattr(settings, "hf_token", "tok")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_extra_body", '{"reasoning_effort": "none"}')
    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "<think></think>ok [S1]"}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers)
        return Resp()

    monkeypatch.setattr(generate.httpx, "post", fake_post)
    out = openai_llm(model="Qwen/Qwen3-8B")("sys", "question")
    assert out.endswith("ok [S1]")  # postprocess() strips the think block later
    assert seen["url"].endswith("/chat/completions")
    assert seen["headers"]["Authorization"] == "Bearer tok"
    assert seen["body"]["reasoning_effort"] == "none" and not seen["body"]["stream"]
    assert seen["body"]["messages"][1]["content"].endswith("/no_think")
    json.dumps(seen["body"])  # serialisable


def test_rate_limit_blocks_after_limit(monkeypatch):
    class Req:
        headers = {"x-forwarded-for": "1.2.3.4, 10.0.0.1"}
        client = None

    monkeypatch.setattr(settings, "rate_limit_per_hour", 2)
    api._hits.clear()
    api.rate_limit(Req())
    api.rate_limit(Req())
    with pytest.raises(HTTPException) as e:
        api.rate_limit(Req())
    assert e.value.status_code == 429
    monkeypatch.setattr(settings, "rate_limit_per_hour", 0)  # off
    api.rate_limit(Req())
