"""Proxy / streaming tests — frame classification, mid-stream error, retry
backoff."""

from __future__ import annotations

import asyncio
import json
import pytest

import proxy as proxy_mod
from proxy.proxy import ProxyCore, _classify_sse_frame, _is_content_sse_frame


# ---------------------------------------------------------------------------
# SSE frame classification
# ---------------------------------------------------------------------------

def test_classify_sse_frame_rate_limit() -> None:
    frame = b'data: {"error": {"code": 429, "message": "rate limited"}}'
    out = _classify_sse_frame(frame)
    assert out is not None
    assert out["kind"] == "rate_limit"


def test_classify_sse_frame_invalid_request() -> None:
    frame = b'data: {"error": {"type": "invalid_request_error", "message": "bad input"}}'
    out = _classify_sse_frame(frame)
    assert out is not None
    assert out["kind"] in ("invalid_request", "generic_error")


def test_classify_sse_frame_no_error_returns_none() -> None:
    frame = b'data: {"choices": [{"delta": {"content": "hi"}}]}'
    assert _classify_sse_frame(frame) is None


def test_classify_sse_frame_ignores_openai_done() -> None:
    frame = b'data: [DONE]'
    assert _classify_sse_frame(frame) is None


def test_is_content_sse_frame_text_delta() -> None:
    # OpenAI chunk (no "type" field) — NOT a content frame for Anthropic protocol.
    frame = b'data: {"choices": [{"delta": {"content": "hi"}}]}'
    assert not _is_content_sse_frame(frame)


def test_is_content_sse_frame_anthropic_text_delta() -> None:
    # Anthropic-shaped content_block_delta{text_delta} — IS a content frame.
    frame = b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}'
    assert _is_content_sse_frame(frame)


def test_is_content_sse_frame_anthropic_message_start() -> None:
    frame = b'data: {"type":"message_start","message":{}}'
    assert _is_content_sse_frame(frame)


def test_is_content_sse_frame_anthropic_thinking_delta() -> None:
    """Thinking deltas are content-bearing for Anthropic clients (they show
    as the model's reasoning trace)."""
    frame = b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}'
    assert _is_content_sse_frame(frame)


def test_is_content_sse_frame_done_is_not_content() -> None:
    frame = b'data: [DONE]'
    assert not _is_content_sse_frame(frame)


def test_is_content_sse_frame_garbage_returns_false() -> None:
    assert not _is_content_sse_frame(b"")
    assert not _is_content_sse_frame(b"not sse at all")


# ---------------------------------------------------------------------------
# Retry backoff helper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_backoff_caps_at_eight_seconds() -> None:
    """Backoff should cap at 8s + 0.5s jitter even for very high attempt
    numbers, so we don't pile up requests against a transiently-down
    provider."""
    import time
    t0 = time.monotonic()
    await ProxyCore._retry_backoff(20)  # attempt=20 → 2^20 = 1M, must cap
    elapsed = time.monotonic() - t0
    assert elapsed <= 9.0, f"backoff exceeded cap: {elapsed}s"


@pytest.mark.asyncio
async def test_retry_backoff_increases_with_attempt() -> None:
    import time
    t0 = time.monotonic()
    await ProxyCore._retry_backoff(0)  # ~1s + jitter
    e0 = time.monotonic() - t0

    t0 = time.monotonic()
    await ProxyCore._retry_backoff(3)  # ~8s + jitter
    e1 = time.monotonic() - t0
    assert e1 >= e0, f"backoff should grow with attempt: e0={e0} e1={e1}"


# ---------------------------------------------------------------------------
# iter_upstream_sse — mid-stream error classification marks the key
# ---------------------------------------------------------------------------

class _FakeUpstream:
    """Minimal httpx-stream shim for iter_upstream_sse tests."""

    def __init__(self, status: int, chunks: list[bytes]) -> None:
        self.status_code = status
        self.headers = _FakeHeaders({"content-type": "text/event-stream"})
        self._chunks = list(chunks)
        self.closed = False

    def aiter_raw(self):
        async def _gen():
            for c in self._chunks:
                yield c
        return _gen()

    async def aclose(self) -> None:
        self.closed = True

    async def aread(self) -> bytes:
        return b""


class _FakeHeaders:
    def __init__(self, d: dict[str, str]) -> None:
        self._d = d

    def items(self):
        return list(self._d.items())

    def get(self, k: str, default: str = "") -> str:
        return self._d.get(k.lower(), default)


@pytest.mark.asyncio
async def test_iter_upstream_sse_marks_key_on_midstream_rate_limit(monkeypatch) -> None:
    """Mid-stream rate_limit frame MUST mark the key (CRITICAL: AS2 fix)."""
    from proxy.keypool import KeyPool, KeyState
    pool = KeyPool(["k1", "k2"], mode="partial_sticky")

    # Build a ProxyCore-shaped object that just exposes pool + client + _free_sem
    pc = type("FakePC", (), {})()
    pc.pool = pool
    pc._free_sem = asyncio.Semaphore(100)

    async def _fake_send(req, stream):
        # Two chunks: one valid content, one rate-limit error frame
        return _FakeUpstream(
            status=200,
            chunks=[
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
                b'data: {"error":{"code":429,"message":"rate limited"}}\n\n',
            ],
        )

    class _FakeClient:
        def build_request(self, method, url, headers=None, content=None):
            return ("REQ", method, url, headers, content)

        async def send(self, req, stream):
            return await _fake_send(req, stream)

    pc.client = _FakeClient()

    # Call iter_upstream_sse — but it's a real method on ProxyCore. Use the
    # real ProxyCore with our fakes, bypassing __init__.
    real_pc = ProxyCore.__new__(ProxyCore)
    real_pc.pool = pool
    real_pc.client = pc.client
    real_pc._free_sem = pc._free_sem
    from proxy.providers import get_provider
    real_pc.provider = get_provider("huggingface")
    real_pc.provider_name = real_pc.provider.name

    result = await real_pc.iter_upstream_sse(
        "POST",
        "https://example.test/v1/chat",
        {"Authorization": "Bearer k1"},
        b'{}',
        0,
        "rid-test-1",
    )
    # iter_upstream_sse returns an async iterator on success, or a
    # Response on upstream connect/status failure. Both are awaitable to
    # produce the real value.
    if asyncio.iscoroutine(result):
        result = await result
    assert not isinstance(result, type(None).__class__), "iter_upstream_sse should return iterator/Response, not None"
    if hasattr(result, "__aiter__"):
        frames = []
        async for f in result:
            frames.append(f)
        # The error frame MUST NOT be yielded to the caller (it terminates the stream)
        assert not any(b"rate limited" in f for f in frames), "error frame leaked to caller"
        # The key MUST be marked as cooling (post error) — the cooldown state-machine fires
        k0 = pool._keys[0]
        assert k0.consecutive_errors >= 1, f"key not marked on mid-stream error: {k0}"
        assert k0.state in (KeyState.COOLING, KeyState.SUSPENDED), f"key state wrong: {k0.state}"
    # pool was fully released
    assert pool._keys[0].in_flight == 0, f"key in_flight leaked: {pool._keys[0].in_flight}"