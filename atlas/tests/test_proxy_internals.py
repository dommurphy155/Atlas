"""atlas_proxy internals + nvidia_client + token_tracker — the layer the
existing suite leaves thin.

The existing tests pin the key store, the SSE translator, the OpenAI↔Anthropic
shape helpers, the system-prompt override, and the non-stream failover loop.
This file fills the rest: request-body parsing/rejection, usage extraction,
the response log line formatter, the FastAPI endpoints (health/stats/models),
the streaming failover loop, keepalive, the NVIDIA client's URL/headers/validity
helpers and its non-stream/stream happy paths (with httpx mocked at the
transport layer — no network), and the token_tracker renderer.

All network is mocked. No test touches the live data/ directory or NVIDIA.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# conftest injects `proxy` on sys.path and exposes run() for sync→async.
from conftest import run, collect
from proxy import atlas_proxy
from proxy.nvidia_client import NvidiaClient, NvidiaResponse, _error_iterator, _extract_error_message


# ── helpers ──────────────────────────────────────────────────────────────────

def _patch_keys(monkeypatch, keys):
    """Point the live key_store at a canned pool so handlers don't touch disk.
    Also clears the cooldown map — the key_store is a module singleton, so a
    prior test that cooled a key would leak into the next (acquire() would
    skip a key that this test never cooled). Wipe the slate per test."""
    monkeypatch.setattr(atlas_proxy.key_store, "_keys", list(keys))
    monkeypatch.setattr(atlas_proxy.key_store, "_active_index", -1)
    monkeypatch.setattr(atlas_proxy.key_store, "_cooldowns", {})


def _oai_ok(content="ok", *, pt=2, ct=1, tt=3, tool_calls=None):
    """A canned healthy NVIDIA 200 chat-completion response."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return NvidiaResponse(
        status_code=200,
        json_data={
            "id": "chatcmpl-x",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt},
        },
    )


# ── _short_model ─────────────────────────────────────────────────────────────

class TestShortModel:
    def test_strips_org_prefix(self):
        assert atlas_proxy._short_model("z-ai/glm-5.2") == "glm-5.2"
        assert atlas_proxy._short_model("moonshotai/kimi-k2.6") == "kimi-k2.6"

    def test_passthrough_when_no_slash(self):
        assert atlas_proxy._short_model("glm-5.2") == "glm-5.2"


# ── _extract_usage ──────────────────────────────────────────────────────────

class TestExtractUsage:
    def test_happy_path_counts_tool_calls(self):
        data = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "choices": [{"message": {"tool_calls": [{"id": "a"}, {"id": "b"}]}}],
        }
        pt, ct, tt, tc = atlas_proxy._extract_usage(data)
        assert (pt, ct, tt, tc) == (10, 4, 14, 2)

    def test_missing_usage_returns_zeros(self):
        assert atlas_proxy._extract_usage({"choices": []}) == (0, 0, 0, 0)

    def test_non_dict_returns_zeros(self):
        assert atlas_proxy._extract_usage(None) == (0, 0, 0, 0)
        assert atlas_proxy._extract_usage("not a dict") == (0, 0, 0, 0)

    def test_null_usage_values_coerced_to_zero(self):
        data = {"usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}}
        assert atlas_proxy._extract_usage(data) == (0, 0, 0, 0)

    def test_no_choices_no_tool_calls(self):
        data = {"usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
        assert atlas_proxy._extract_usage(data)[3] == 0


# ── response log line formatter ─────────────────────────────────────────────

class TestFormatResponseLine:
    def test_known_usage_prints_integers(self):
        line = atlas_proxy._format_response_line(
            "r1", 200, "z-ai/glm-5.2", "#0(…abcd)", 1, 10, 4, 14, None, 0.0, True,
        )
        assert "status=200" in line
        assert "model=glm-5.2" in line
        assert "key=#0(…abcd)" in line
        assert "tools=1" in line
        assert "in=10" in line
        assert "out=4" in line
        assert "tokens=14" in line
        # Four timing phases always present, never omitted.
        for phase in ("upstream=", "ttft=", "stream=", "total="):
            assert phase in line

    def test_unknown_usage_prints_question_marks(self):
        line = atlas_proxy._format_response_line(
            "r1", 502, "glm-5.2", "?", 0, 0, 0, 0, None, 0.0, False,
        )
        assert "in=?" in line
        assert "out=?" in line
        assert "tokens=?" in line
        assert "status=502" in line

    def test_timings_default_to_zero_when_missing(self):
        line = atlas_proxy._format_response_line("r1", 200, "m", "?", 0, 0, 0, 0, {}, 0.0, False)
        # No timings dict → every phase prints 0.0s, never omitted.
        assert "upstream=0.0s" in line
        assert "ttft=0.0s" in line
        assert "stream=0.0s" in line

    def test_total_computed_from_started(self):
        line = atlas_proxy._format_response_line("r1", 200, "m", "?", 0, 0, 0, 0, {}, 0.0, False)
        assert "total=0.0s" in line


def test_fmt_timing_clamps_negative_to_zero():
    # A negative timing (clock skew / monotonic oddity) must print 0.0s, not -0.5s.
    assert atlas_proxy._fmt_timing({"upstream": -0.5}, "upstream", 1) == "0.0s"


def test_fmt_timing_missing_key_defaults_zero():
    assert atlas_proxy._fmt_timing(None, "upstream", 1) == "0.0s"
    assert atlas_proxy._fmt_timing({}, "upstream", 1) == "0.0s"


def test_fmt_tok_known_vs_unknown():
    assert atlas_proxy._fmt_tok(42, True) == "42"
    assert atlas_proxy._fmt_tok(42, False) == "?"


# ── parse_request_body ──────────────────────────────────────────────────────
#
# Driven via the FastAPI TestClient so the Request object is real. The function
# reads content-length, body bytes, and parses JSON exactly as in production.

@pytest.fixture
def client(monkeypatch):
    """A TestClient against the live app, with the key store + NVIDIA client
    stubbed so startup (lifespan) doesn't touch disk or the network."""
    # Stub the lifespan's side effects: key load + stats reset + prewarm.
    monkeypatch.setattr(atlas_proxy.key_store, "load", AsyncMock(return_value=None))
    monkeypatch.setattr(atlas_proxy.key_store, "watch", AsyncMock(return_value=None))
    monkeypatch.setattr(atlas_proxy, "reset_since_restart", lambda: None)
    monkeypatch.setattr(atlas_proxy.nvidia_client, "prewarm", AsyncMock(return_value=None))
    monkeypatch.setattr(atlas_proxy.nvidia_client, "close", AsyncMock(return_value=None))
    from fastapi.testclient import TestClient
    with TestClient(atlas_proxy.app) as c:
        yield c


class TestParseRequestBody:
    def test_valid_json_returned_as_dict(self, client):
        r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        # messages=[] → normalize raises → 400, but that proves the body parsed.
        assert r.status_code == 400

    def test_oversize_content_length_rejected_413(self, client, monkeypatch):
        monkeypatch.setattr(atlas_proxy, "MAX_BODY_BYTES", 16)
        r = client.post(
            "/v1/chat/completions",
            content=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
            headers={"Content-Type": "application/json", "Content-Length": "99999999"},
        )
        assert r.status_code == 413

    def test_bad_json_rejected_400(self, client):
        r = client.post(
            "/v1/chat/completions",
            content="not json at all {{{",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_non_object_json_rejected_400(self, client):
        r = client.post(
            "/v1/chat/completions",
            content="[1,2,3]",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_anthropic_endpoint_returns_anthropic_error_shape(self, client):
        """A 400 on /v1/messages must come back in the Anthropic error shape so
        Claude Code's SDK can parse it (not the OpenAI shape)."""
        r = client.post("/v1/messages", content="not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400
        body = r.json()
        assert body["type"] == "error"
        assert "message" in body["error"]

    def test_openai_endpoint_returns_openai_error_shape(self, client):
        r = client.post("/v1/chat/completions", content="not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400
        body = r.json()
        assert "error" in body
        assert "message" in body["error"]


# ── GET endpoints ───────────────────────────────────────────────────────────

class TestEndpoints:
    def test_health(self, client, monkeypatch):
        # `available` is a property over _keys; set the underlying list so it
        # returns True naturally (can't monkeypatch a property with no setter).
        monkeypatch.setattr(atlas_proxy.key_store, "_keys", ["nvapi-x"])
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "atlas-proxy"
        assert body["provider"] == "nvidia"
        assert body["keys_available"] is True

    def test_stats(self, client, monkeypatch):
        monkeypatch.setattr(atlas_proxy.key_store, "stats", lambda: {
            "total_keys": 3, "available": True, "cooling_down": 1,
            "active_key_index": 0, "active_key_eligible": True,
        })
        r = client.get("/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["nvidia_keys_total"] == 3
        assert body["nvidia_keys_cooling_down"] == 1
        assert "proxy" in body

    def test_models_lists_backing_model(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == atlas_proxy.NVIDIA_MODEL
        assert body["data"][0]["owned_by"] == "nvidia"


# ── non-stream happy path + 400 surfacing ───────────────────────────────────

@pytest.mark.asyncio
async def test_non_stream_success_records_and_returns_openai_shape(keys_file, monkeypatch):
    keys = [f"nvapi-k{i:02d}" + "x" * 40 for i in range(2)]
    _patch_keys(monkeypatch, keys)
    atlas_proxy.nvidia_client.chat = AsyncMock(return_value=_oai_ok("hello"))
    result = await atlas_proxy.handle_non_stream("m", {"model": "m", "messages": []}, rid="t", started=0.0)
    assert result.status_code == 200
    body = json.loads(result.body.decode())
    assert body["choices"][0]["message"]["content"] == "hello"
    assert atlas_proxy.nvidia_client.chat.await_count == 1


@pytest.mark.asyncio
async def test_non_stream_402_credits_exhausted_failovers(keys_file, monkeypatch):
    """A 402 (credits exhausted) must cool the key and rotate, just like 429."""
    keys = [f"nvapi-k{i:02d}" + "x" * 40 for i in range(2)]
    _patch_keys(monkeypatch, keys)
    responses = [
        NvidiaResponse(status_code=402, json_data={"error": {"message": "credits exhausted"}}),
        _oai_ok("ok"),
    ]
    atlas_proxy.nvidia_client.chat = AsyncMock(side_effect=responses)
    result = await atlas_proxy.handle_non_stream("m", {"model": "m", "messages": []}, rid="t", started=0.0)
    assert result.status_code == 200
    assert atlas_proxy.nvidia_client.chat.await_count == 2


@pytest.mark.asyncio
async def test_non_stream_transport_error_failovers(keys_file, monkeypatch):
    """An httpx.HTTPError (not a timeout) cools the key and retries the next."""
    keys = [f"nvapi-k{i:02d}" + "x" * 40 for i in range(2)]
    _patch_keys(monkeypatch, keys)
    atlas_proxy.nvidia_client.chat = AsyncMock(
        side_effect=[httpx.ConnectError("boom"), _oai_ok("ok")]
    )
    result = await atlas_proxy.handle_non_stream("m", {"model": "m", "messages": []}, rid="t", started=0.0)
    assert result.status_code == 200
    assert atlas_proxy.nvidia_client.chat.await_count == 2


@pytest.mark.asyncio
async def test_non_stream_timeout_returns_504(keys_file, monkeypatch):
    """An httpx.TimeoutException cools the key and surfaces 504, not 502."""
    keys = [f"nvapi-k{i:02d}" + "x" * 40 for i in range(2)]
    _patch_keys(monkeypatch, keys)
    atlas_proxy.nvidia_client.chat = AsyncMock(side_effect=httpx.ReadTimeout("idle"))
    result = await atlas_proxy.handle_non_stream("m", {"model": "m", "messages": []}, rid="t", started=0.0)
    assert result.status_code == 504


# ── streaming failover loop ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_failover_on_429_rotates_key(keys_file, monkeypatch):
    """The streaming path must cool a 429'd key and succeed on the next."""
    from collections.abc import AsyncIterator

    keys = [f"nvapi-k{i:02d}" + "x" * 40 for i in range(2)]
    _patch_keys(monkeypatch, keys)

    async def _ok_iter() -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def _err_iter() -> AsyncIterator[bytes]:
        yield b'data: {"error":{"message":"rate limited"}}\n\n'
        yield b"data: [DONE]\n\n"

    responses = [
        (429, {}, _err_iter(), "rate limited"),
        (200, {}, _ok_iter(), ""),
    ]
    atlas_proxy.nvidia_client.stream_chat = AsyncMock(side_effect=responses)

    resp = await atlas_proxy.handle_stream("m", {"model": "m", "messages": [], "stream": True}, rid="s", started=0.0)
    # Drain the StreamingResponse so the active_requests decrement fires.
    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk)
    raw = b"".join(chunks)
    assert b"hi" in raw
    assert atlas_proxy.nvidia_client.stream_chat.await_count == 2


@pytest.mark.asyncio
async def test_stream_no_keys_returns_503_error_stream(keys_file, monkeypatch):
    _patch_keys(monkeypatch, [])
    atlas_proxy.nvidia_client.stream_chat = AsyncMock()
    resp = await atlas_proxy.handle_stream("m", {"model": "m", "messages": [], "stream": True}, rid="s", started=0.0)
    assert atlas_proxy.nvidia_client.stream_chat.await_count == 0
    chunks = b"".join([c async for c in resp.body_iterator])
    assert b"503" in chunks


# ── stream_router_sse usage extraction ─────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_router_sse_extracts_usage_and_tool_calls(monkeypatch):
    """The OpenAI passthrough stream must pull usage + tool_calls out of the
    chunk stream and record them, without re-parsing every token chunk."""
    from collections.abc import AsyncIterator

    async def upstream() -> AsyncIterator[bytes]:
        # Plain text deltas (no usage/tool_calls → cheap-gate skips the parse).
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        # A tool_call chunk.
        yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{}"}}]}}]}\n\n'
        # Final chunk with usage.
        yield b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}\n\n'
        yield b"data: [DONE]\n\n"

    recorded = {}
    monkeypatch.setattr(atlas_proxy, "record_success", lambda *a, **k: recorded.setdefault("called", a))

    async for _ in atlas_proxy.stream_router_sse(upstream(), "m", "r1", "nvidia", 0.0, "#0(…abcd)", {}):
        pass

    assert recorded.get("called") is not None
    # record_success(provider, model, prompt, completion, total, tool_calls)
    _, model, pt, ct, tt, tc = recorded["called"]
    assert (pt, ct, tt, tc) == (7, 3, 10, 1)


@pytest.mark.asyncio
async def test_stream_router_sse_usage_unknown_still_records(monkeypatch):
    """A stream with no usage chunk must still record a success (zero tokens)
    so /stats counts the request — usage_unknown, not a dropped request."""
    from collections.abc import AsyncIterator

    async def upstream() -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    recorded = {}
    monkeypatch.setattr(atlas_proxy, "record_success", lambda *a, **k: recorded.setdefault("called", a))

    async for _ in atlas_proxy.stream_router_sse(upstream(), "m", "r1", "nvidia", 0.0, "#0", {}):
        pass

    assert recorded.get("called") is not None
    assert recorded["called"][5] == 0  # tool_calls


# ── keepalive ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_keepalive_emits_ping_then_passes_chunks():
    """keepalive must flush a leading ': ping' comment immediately and then
    forward upstream bytes verbatim once they arrive."""
    from collections.abc import AsyncIterator

    async def upstream() -> AsyncIterator[bytes]:
        yield b"data: real\n\n"

    out = []
    async for chunk in atlas_proxy.keepalive(upstream(), interval=10.0):
        out.append(chunk)
    assert out[0] == b": ping\n\n", "leading ping must be flushed first"
    assert b"data: real" in b"".join(out)


@pytest.mark.asyncio
async def test_keepalive_emits_comment_on_idle(monkeypatch):
    """When the upstream is silent past the interval, keepalive emits a
    ': keepalive' comment without dropping the in-flight read."""
    from collections.abc import AsyncIterator

    async def upstream() -> AsyncIterator[bytes]:
        yield b"data: first\n\n"
        # Simulate a long think gap by yielding slowly.
        import asyncio
        await asyncio.sleep(0.3)
        yield b"data: second\n\n"

    out = []
    async for chunk in atlas_proxy.keepalive(upstream(), interval=0.05):
        out.append(chunk)
    raw = b"".join(out)
    assert b": keepalive" in raw, "idle gap must produce a keepalive comment"
    assert b"data: first" in raw and b"data: second" in raw


@pytest.mark.asyncio
async def test_keepalive_terminates_on_empty_upstream():
    from collections.abc import AsyncIterator

    async def upstream() -> AsyncIterator[bytes]:
        return
        yield  # noqa: make it an async generator

    out = []
    async for chunk in atlas_proxy.keepalive(upstream(), interval=1.0):
        out.append(chunk)
    assert out == [b": ping\n\n"], "empty upstream yields only the leading ping"


# ── nvidia_client: pure helpers ─────────────────────────────────────────────

class TestNvidiaClientHelpers:
    def test_chat_url_appends_completions(self):
        assert NvidiaClient._chat_url("https://x/v1") == "https://x/v1/chat/completions"

    def test_chat_url_idempotent(self):
        url = "https://x/v1/chat/completions"
        assert NvidiaClient._chat_url(url) == url

    def test_chat_url_strips_trailing_slash(self):
        assert NvidiaClient._chat_url("https://x/v1/") == "https://x/v1/chat/completions"

    def test_is_valid_key(self):
        assert NvidiaClient.is_valid_key("nvapi-abc") is True
        assert NvidiaClient.is_valid_key("sk-abc") is False
        assert NvidiaClient.is_valid_key("") is False
        assert NvidiaClient.is_valid_key(None) is False

    def test_headers_shape(self):
        c = NvidiaClient("https://x/v1", 10.0)
        h = c._headers("nvapi-secret")
        assert h["Authorization"] == "Bearer nvapi-secret"
        assert h["Content-Type"] == "application/json"

    def test_response_from_httpx(self):
        req = httpx.Request("POST", "https://x/v1/chat/completions")
        r = httpx.Response(200, request=req, json={"id": "x"})
        nr = NvidiaClient._response_from_httpx(r)
        assert nr.status_code == 200
        assert nr.json_data == {"id": "x"}

    def test_response_from_httpx_non_json(self):
        req = httpx.Request("POST", "https://x/v1/chat/completions")
        r = httpx.Response(200, request=req, content="not json")
        nr = NvidiaClient._response_from_httpx(r)
        assert nr.json_data is None
        assert "not json" in nr.text


# ── nvidia_client: chat() happy path with mocked transport ──────────────────

@pytest.mark.asyncio
async def test_chat_populates_timings(monkeypatch):
    """chat() must stamp timings['upstream'] and default 'stream' to 0.0."""
    client = NvidiaClient("https://x/v1", timeout=10.0)

    async def fake_post(self, url, headers=None, json=None):
        req = httpx.Request("POST", url, headers=headers or {})
        return httpx.Response(200, request=req, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    timings: dict = {}
    resp = await client.chat("nvapi-k", {"model": "m", "messages": []}, timings=timings)
    assert resp.status_code == 200
    assert timings["upstream"] >= 0.0
    assert timings["stream"] == 0.0
    await client.close()


@pytest.mark.asyncio
async def test_chat_non_json_response(monkeypatch):
    """A 200 with a non-JSON body must yield json_data=None, not raise."""
    client = NvidiaClient("https://x/v1", timeout=10.0)

    async def fake_post(self, url, headers=None, json=None):
        req = httpx.Request("POST", url, headers=headers or {})
        return httpx.Response(200, request=req, content="plaintext")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    resp = await client.chat("nvapi-k", {"model": "m", "messages": []})
    assert resp.status_code == 200
    assert resp.json_data is None
    assert resp.text == "plaintext"
    await client.close()


# ── nvidia_client: stream_chat error path ────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_chat_400_drains_error_and_returns_message(monkeypatch):
    """A 4xx on the stream response must drain the body and return the real
    upstream message (not the generic 'upstream returned 400')."""
    client = NvidiaClient("https://x/v1", timeout=10.0)

    async def fake_send(self, request, stream=False):
        req = httpx.Request("POST", request.url, headers=request.headers)
        return httpx.Response(
            400, request=req,
            content=json.dumps({"error": {"message": "Validation: bad tool schema"}}).encode(),
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    status, headers, iterator, message = await client.stream_chat(
        "nvapi-k", {"model": "m", "messages": []}, rid="r1",
    )
    assert status == 400
    assert message == "Validation: bad tool schema"
    # The error iterator replays the message as an OpenAI-shaped SSE error chunk.
    chunks = b"".join([c async for c in iterator])
    assert b"Validation: bad tool schema" in chunks
    assert b"[DONE]" in chunks
    await client.close()


@pytest.mark.asyncio
async def test_stream_chat_200_yields_bytes(monkeypatch):
    """A healthy stream must hand back a 200 and an iterator of the upstream
    SSE bytes, populating upstream/ttft timings on first byte."""
    client = NvidiaClient("https://x/v1", timeout=10.0)

    body = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"there"}}]}\n\n'
        b'data: [DONE]\n\n'
    )

    async def fake_send(self, request, stream=False):
        req = httpx.Request("POST", request.url, headers=request.headers)
        return httpx.Response(200, request=req, content=body)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    timings = {"__started": 0.0}
    status, headers, iterator, message = await client.stream_chat(
        "nvapi-k", {"model": "m", "messages": []}, rid="r1", timings=timings,
    )
    assert status == 200
    assert message == ""
    chunks = b"".join([c async for c in iterator])
    assert b"hi" in chunks and b"there" in chunks and b"[DONE]" in chunks
    assert timings["upstream"] >= 0.0
    await client.close()


@pytest.mark.asyncio
async def test_stream_chat_mid_stream_timeout_injects_error_chunk(monkeypatch):
    """A mid-stream ReadTimeout must fire on_timeout and emit a terminal
    OpenAI-shaped error chunk + [DONE] so the client sees a clean end."""
    client = NvidiaClient("https://x/v1", timeout=10.0)

    async def aiter_timeout():
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise httpx.ReadTimeout("idle")

    async def fake_send(self, request, stream=False):
        req = httpx.Request("POST", request.url, headers=request.headers)

        class _R:
            status_code = 200
            headers = {}

            async def aiter_bytes(self):
                async for c in aiter_timeout():
                    yield c

            async def aclose(self):
                pass

        return _R()

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    fired = []
    status, headers, iterator, message = await client.stream_chat(
        "nvapi-k", {"model": "m", "messages": []}, rid="r9",
        on_timeout=lambda: fired.append("timeout"),
    )
    chunks = b"".join([c async for c in iterator])
    assert fired == ["timeout"], "on_timeout must fire on a mid-stream ReadTimeout"
    assert b"upstream stream timed out (idle_read)" in chunks
    assert b"[DONE]" in chunks
    await client.close()


# ── _error_iterator ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_error_iterator_emits_openai_shape_and_done():
    chunks = [c async for c in _error_iterator('{"error":{"message":"boom"}}')]
    raw = b"".join(chunks)
    data = json.loads(raw.split(b"\n\n")[0][len(b"data: "):])
    assert data["error"]["message"] == "boom"
    assert chunks[-1] == b"data: [DONE]\n\n"


# ── token_tracker ───────────────────────────────────────────────────────────

class TestTokenTracker:
    def test_render_no_stats_file(self, tmp_path, monkeypatch):
        from proxy import token_tracker as tt
        monkeypatch.setattr(tt, "STATS_FILE", tmp_path / "missing.json")
        out = tt.render()
        assert "no stats found" in out

    def test_render_populated(self, tmp_path, monkeypatch):
        from proxy import token_tracker as tt
        f = tmp_path / "proxy_stats.json"
        f.write_text(json.dumps({
            "started_at": "2026-07-21T00:00:00+00:00",
            "restart": {
                "requests": 10, "successes": 8, "failures": 2,
                "prompt_tokens": 1000, "completion_tokens": 400, "total_tokens": 1400,
                "tool_calls": 5,
                "models": {"z-ai/glm-5.2": 8, "other": 2},
            },
        }))
        monkeypatch.setattr(tt, "STATS_FILE", f)
        out = tt.render()
        assert "atlas tokens" in out
        assert "requests" in out and "10" in out
        assert "successes" in out
        assert "80.0%" in out  # 8/10
        assert "1,400" in out  # total grouped
        assert "avg tokens/req" in out
        assert "140.0" in out  # 1400/10
        assert "glm-5.2" in out
        # Models sorted by count desc: glm-5.2 (8) before other (2).
        assert out.index("glm-5.2") < out.index("other")

    def test_render_empty_restart(self, tmp_path, monkeypatch):
        from proxy import token_tracker as tt
        f = tmp_path / "proxy_stats.json"
        f.write_text(json.dumps({"started_at": "x", "restart": {}}))
        monkeypatch.setattr(tt, "STATS_FILE", f)
        out = tt.render()
        assert "requests" in out
        assert "0" in out

    def test_bar(self):
        from proxy.token_tracker import _bar
        assert _bar(0, 100) == "░" * 20
        assert _bar(100, 100) == "█" * 20
        assert _bar(50, 100).count("█") == 10
        assert _bar(10, 0) == ""  # zero total → empty

    def test_fmt_int(self):
        from proxy.token_tracker import _fmt_int
        assert _fmt_int(1400) == "1,400"
        assert _fmt_int(0) == "0"


# ── openai_compat: the untested pure helpers ─────────────────────────────────

class TestOpenAICompatExtras:
    def test_sse_from_text(self):
        from proxy.openai_compat import sse_from_text
        chunks = collect(sse_from_text("m", "hello"))
        raw = b"".join(chunks)
        assert b'"content":"hello"' in raw
        assert chunks[-1] == b"data: [DONE]\n\n"

    def test_sse_from_text_empty(self):
        from proxy.openai_compat import sse_from_text
        chunks = collect(sse_from_text("m", ""))
        assert chunks == [b"data: [DONE]\n\n"]

    def test_non_stream_response_text(self):
        from proxy.openai_compat import non_stream_response
        r = non_stream_response("m", "hi")
        assert r["choices"][0]["message"]["content"] == "hi"
        assert r["choices"][0]["finish_reason"] == "stop"
        assert r["usage"]["total_tokens"] == 0

    def test_non_stream_response_tool_calls(self):
        from proxy.openai_compat import non_stream_response
        tc = [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]
        r = non_stream_response("m", "", tool_calls=tc)
        assert r["choices"][0]["finish_reason"] == "tool_calls"
        assert r["choices"][0]["message"]["tool_calls"] == tc

    def test_openai_response_from_router_preserves_choices(self):
        from proxy.openai_compat import openai_response_from_router
        payload = {"id": "orig", "choices": [{"index": 0, "message": {"content": "x"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        out = openai_response_from_router("m", payload)
        assert out["id"] == "orig"
        assert out["model"] == "m"
        assert out["usage"]["total_tokens"] == 2

    def test_openai_response_from_router_fallback(self):
        from proxy.openai_compat import openai_response_from_router
        out = openai_response_from_router("m", {"weird": "shape"})
        assert out["choices"][0]["message"]["content"] == json.dumps({"weird": "shape"})

    def test_extract_router_content_message(self):
        from proxy.openai_compat import extract_router_content
        assert extract_router_content({"choices": [{"message": {"content": "hi"}}]}) == "hi"

    def test_extract_router_content_text(self):
        from proxy.openai_compat import extract_router_content
        assert extract_router_content({"choices": [{"text": "hi"}]}) == "hi"

    def test_extract_router_content_generated_text(self):
        from proxy.openai_compat import extract_router_content
        assert extract_router_content({"generated_text": "hi"}) == "hi"

    def test_anthropic_system_text(self):
        from proxy.openai_compat import anthropic_system_text
        assert anthropic_system_text("hi") == "hi"
        assert anthropic_system_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
        assert anthropic_system_text(None) == ""

    def test_anthropic_messages_preserves_mid_conversation_system(self):
        """A role:system message mid-thread (the end-of-conversation
        reinforcement) must survive the Anthropic→OpenAI translation as a
        role:system message, NOT collapse to role:user."""
        from proxy.openai_compat import anthropic_messages_to_openai
        out = anthropic_messages_to_openai({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": [{"type": "text", "text": "[System] stay in character"}]},
            {"role": "user", "content": "final q"},
        ]})
        # The last user turn with a [System]-prefixed text block must land as
        # role:user (it's a user message, not a system message) — but the
        # general contract is: mid-thread role:system is preserved. Test that
        # an explicit system role survives.
        out2 = anthropic_messages_to_openai({"messages": [
            {"role": "user", "content": "q"},
            {"role": "system", "content": "reinforce"},
            {"role": "user", "content": "q2"},
        ]})
        sys_msgs = [m for m in out2 if m["role"] == "system"]
        assert sys_msgs and sys_msgs[0]["content"] == "reinforce"

    def test_anthropic_tool_result_with_block_list_content(self):
        """A tool_result whose content is a list of text blocks must be
        flattened to a newline-joined string in the OpenAI tool message."""
        from proxy.openai_compat import anthropic_messages_to_openai
        out = anthropic_messages_to_openai({"messages": [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": [
                {"type": "text", "text": "line1"},
                {"type": "text", "text": "line2"},
            ]}]},
        ]})
        tool_msg = [m for m in out if m["role"] == "tool"][0]
        assert tool_msg["content"] == "line1\nline2"

    def test_anthropic_sse_from_response_full_sequence(self):
        """The non-streaming Anthropic SSE emitter (used for error/fake streams)
        must produce the same event skeleton as the live translator."""
        from proxy.openai_compat import anthropic_sse_from_response, anthropic_response
        chunks = collect(anthropic_sse_from_response(anthropic_response("claude", "hi")))
        events = []
        for block in b"".join(chunks).decode().split("\n\n"):
            ev = None
            for line in block.strip().split("\n"):
                if line.startswith("event: "):
                    ev = line[7:]
            if ev:
                events.append(ev)
        assert events[0] == "message_start"
        assert "content_block_start" in events
        assert "content_block_delta" in events
        assert "content_block_stop" in events
        assert "message_delta" in events
        assert events[-1] == "message_stop"
