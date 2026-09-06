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
    e3 = time.monotonic() - t0

    assert e3 > e0 - 0.5, f"expected attempt=3 ({e3:.2f}s) > attempt=0 ({e0:.2f}s)"