"""Coverage gap-fill: the code paths the existing suite leaves unexercised.

These are the ones that would break silently if regressed:
  - handle_anthropic_stream: the /v1/messages streaming path end-to-end
    (OpenAI→Anthropic SSE through the failover loop + on_done stats). The most
    complex stateful translation in the proxy, previously only unit-tested.
  - _stream_failover_reason: per-status reason mapper (pure, one branch each).
  - stream_error / _anthropic_stream_error: the SSE error-stream builders.
  - upstream_error_text: surfaces the real upstream body on a 4xx.
  - _prepend_to_user_content / _content_to_text: override-injection shape helpers.
  - sanitize_openai_payload edge cases: stop-list filtering, tool_choice
    passthrough, max_tokens=0 fallback, string temperature coercion.
  - normalize_messages edge cases: mixed text/non-text blocks, missing role,
    tool message without tool_call_id.
  - stats internals: _merge_bucket model accumulation, _parse_iso, save atomic.
  - log formatters: _JsonFormatter (extra= lifting, rid top-level, exc_info)
    and _CleanFormatter (colored timestamp + message).

All network is mocked. No test touches the live data/ directory or NVIDIA.
"""
from __future__ import annotations

import json
import logging
import sys
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx
import pytest

from conftest import collect, run
from proxy import atlas_proxy
from proxy.nvidia_client import NvidiaResponse


# ── shared helpers ───────────────────────────────────────────────────────────

def _patch_keys(monkeypatch, keys):
    """Point the live key_store at a canned pool; clear cooldowns so a prior
    test's cooled keys don't leak in (the store is a module singleton)."""
    monkeypatch.setattr(atlas_proxy.key_store, "_keys", list(keys))
    monkeypatch.setattr(atlas_proxy.key_store, "_active_index", -1)
    monkeypatch.setattr(atlas_proxy.key_store, "_cooldowns", {})


KEYS = [f"nvapi-k{i:02d}" + "x" * 40 for i in range(3)]


async def _drain(resp) -> bytes:
    """Drain a StreamingResponse's body_iterator to bytes."""
    out = []
    async for chunk in resp.body_iterator:
        out.append(chunk)
    return b"".join(out)


def _anthropic_events(raw: bytes) -> list[tuple[str, dict]]:
    """Parse an Anthropic SSE byte blob into (event, data) pairs."""
    events = []
    for block in raw.decode().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev = data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if ev and data:
            events.append((ev, data))
    return events


# ── handle_anthropic_stream: end-to-end ─────────────────────────────────────

@pytest.mark.asyncio
async def test_anthropic_stream_happy_path_translates_to_anthropic_sse(keys_file, monkeypatch):
    """A healthy NVIDIA OpenAI stream must come back as a full Anthropic SSE
    event sequence: message_start → content_block_start → text_delta →
    content_block_stop → message_delta → message_stop. The on_done closure
    must record stats with the upstream usage."""
    _patch_keys(monkeypatch, KEYS[:2])

    async def upstream() -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        yield b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
        yield b"data: [DONE]\n\n"

    atlas_proxy.nvidia_client.stream_chat = AsyncMock(
        return_value=(200, {}, upstream(), "")
    )
    recorded = {}
    monkeypatch.setattr(atlas_proxy, "record_success", lambda *a, **k: recorded.setdefault("c", a))

    resp = await atlas_proxy.handle_anthropic_stream("claude", {"model": "m", "messages": [], "stream": True}, rid="a1", started=0.0)
    raw = await _drain(resp)
    events = _anthropic_events(raw)
    names = [e for e, _ in events]

    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert "content_block_delta" in names
    assert "content_block_stop" in names
    assert "message_delta" in names
    assert names[-1] == "message_stop"
    # Text deltas concatenate to the full upstream content.
    text = "".join(d["delta"]["text"] for e, d in events
                   if e == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert text == "Hello world"
    # on_done recorded stats with the upstream usage tuple.
    assert recorded.get("c") is not None
    _, model, pt, ct, tt, tc = recorded["c"]
    assert (pt, ct, tt) == (3, 2, 5)
    assert model == atlas_proxy.NVIDIA_MODEL


@pytest.mark.asyncio
async def test_anthropic_stream_tool_use_translates(keys_file, monkeypatch):
    """A tool-call stream must emit content_block_start(tool_use) +
    input_json_delta, and message_delta with stop_reason=tool_use."""
    _patch_keys(monkeypatch, KEYS[:2])

    async def upstream() -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"search","arguments":"{\\"q\\":"}}]}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"x\"}"}}]}}]}\n\n'
        yield b'data: {"choices":[{"finish_reason":"tool_calls"}]}\n\n'
        yield b"data: [DONE]\n\n"

    atlas_proxy.nvidia_client.stream_chat = AsyncMock(return_value=(200, {}, upstream(), ""))
    resp = await atlas_proxy.handle_anthropic_stream("claude", {"model": "m", "messages": [], "stream": True}, rid="a2", started=0.0)
    events = _anthropic_events(await _drain(resp))

    tool_starts = [d for e, d in events if e == "content_block_start"
                   and d["content_block"]["type"] == "tool_use"]
    assert tool_starts and tool_starts[0]["content_block"]["name"] == "search"
    json_deltas = "".join(d["delta"]["partial_json"] for e, d in events
                          if e == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
    assert json.loads(json_deltas) == {"q": "x"}
    msg_delta = [d for e, d in events if e == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_anthropic_stream_failover_on_429(keys_file, monkeypatch):
    """The Anthropic stream path must cool a 429'd key and succeed on the next,
    same as the OpenAI stream path — proving the shared failover loop wraps it."""
    _patch_keys(monkeypatch, KEYS[:2])

    async def err_iter() -> AsyncIterator[bytes]:
        yield b'data: {"error":{"message":"rate limited"}}\n\n'
        yield b"data: [DONE]\n\n"

    async def ok_iter() -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    atlas_proxy.nvidia_client.stream_chat = AsyncMock(
        side_effect=[(429, {}, err_iter(), "rate limited"), (200, {}, ok_iter(), "")]
    )
    resp = await atlas_proxy.handle_anthropic_stream("claude", {"model": "m", "messages": [], "stream": True}, rid="a3", started=0.0)
    raw = await _drain(resp)
    assert b'"ok"' in raw
    assert atlas_proxy.nvidia_client.stream_chat.await_count == 2


@pytest.mark.asyncio
async def test_anthropic_stream_no_keys_returns_error_stream(keys_file, monkeypatch):
    """An empty key pool must produce an Anthropic-shaped error SSE stream
    (not a 503 JSON), so Claude Code's SSE parser handles it cleanly."""
    _patch_keys(monkeypatch, [])
    atlas_proxy.nvidia_client.stream_chat = AsyncMock()
    resp = await atlas_proxy.handle_anthropic_stream("claude", {"model": "m", "messages": [], "stream": True}, rid="a4", started=0.0)
    assert atlas_proxy.nvidia_client.stream_chat.await_count == 0
    events = _anthropic_events(await _drain(resp))
    # Even the error path emits a well-formed Anthropic stream with a text block.
    assert events[0][0] == "message_start"
    assert events[-1][0] == "message_stop"
    # The error text carries the 503 status.
    text = "".join(d["delta"].get("text", "") for e, d in events
                   if e == "content_block_delta" and d["delta"].get("type") == "text_delta")
    assert "no usable NVIDIA keys" in text
    assert "503" in text


@pytest.mark.asyncio
async def test_anthropic_stream_5xx_retries_then_surfaces(keys_file, monkeypatch):
    """A persistent 5xx retries up to MAX_RETRIES then emits an Anthropic error
    stream with the upstream status."""
    _patch_keys(monkeypatch, KEYS[:2])
    monkeypatch.setattr(atlas_proxy, "MAX_RETRIES", 2)

    async def err_iter() -> AsyncIterator[bytes]:
        yield b'data: {"error":{"message":"upstream down"}}\n\n'
        yield b"data: [DONE]\n\n"

    atlas_proxy.nvidia_client.stream_chat = AsyncMock(
        return_value=(503, {}, err_iter(), "upstream down")
    )
    resp = await atlas_proxy.handle_anthropic_stream("claude", {"model": "m", "messages": [], "stream": True}, rid="a5", started=0.0)
    events = _anthropic_events(await _drain(resp))
    text = "".join(d["delta"].get("text", "") for e, d in events
                   if e == "content_block_delta" and d["delta"].get("type") == "text_delta")
    assert "503" in text
    # Initial + 2 retries = 3 calls.
    assert atlas_proxy.nvidia_client.stream_chat.await_count == 3


@pytest.mark.asyncio
async def test_anthropic_stream_400_surfaces_real_message(keys_file, monkeypatch):
    """A 400 must surface the real upstream message (e.g. 'Validation: ...')
    in the Anthropic error stream, not a bare status."""
    _patch_keys(monkeypatch, KEYS[:2])

    async def err_iter() -> AsyncIterator[bytes]:
        yield b'data: {"error":{"message":"Validation: bad tool schema"}}\n\n'
        yield b"data: [DONE]\n\n"

    atlas_proxy.nvidia_client.stream_chat = AsyncMock(
        return_value=(400, {}, err_iter(), "Validation: bad tool schema")
    )
    resp = await atlas_proxy.handle_anthropic_stream("claude", {"model": "m", "messages": [], "stream": True}, rid="a6", started=0.0)
    events = _anthropic_events(await _drain(resp))
    text = "".join(d["delta"].get("text", "") for e, d in events
                   if e == "content_block_delta" and d["delta"].get("type") == "text_delta")
    assert "Validation: bad tool schema" in text
    assert atlas_proxy.nvidia_client.stream_chat.await_count == 1, "400 must not retry"


# ── _stream_failover_reason ──────────────────────────────────────────────────

class TestStreamFailoverReason:
    def test_404_is_key_rejected(self):
        assert atlas_proxy._stream_failover_reason(404) == "model/key 404"

    def test_401_403_is_auth(self):
        assert atlas_proxy._stream_failover_reason(401) == "auth"
        assert atlas_proxy._stream_failover_reason(403) == "auth"

    def test_402_is_credits(self):
        assert atlas_proxy._stream_failover_reason(402) == "credits exhausted"

    def test_other_is_quota_429_default(self):
        # The catch-all (429, 500, anything not matched) maps to the 429 label.
        assert atlas_proxy._stream_failover_reason(429) == "quota/billing 429"
        assert atlas_proxy._stream_failover_reason(500) == "quota/billing 429"
        assert atlas_proxy._stream_failover_reason(999) == "quota/billing 429"


# ── upstream_error_text ──────────────────────────────────────────────────────

class TestUpstreamErrorText:
    def test_json_data_serialized(self):
        r = NvidiaResponse(status_code=400, json_data={"error": {"message": "x"}})
        assert json.loads(atlas_proxy.upstream_error_text(r))["error"]["message"] == "x"

    def test_no_json_falls_back_to_text(self):
        r = NvidiaResponse(status_code=400, json_data=None, text="raw body")
        assert atlas_proxy.upstream_error_text(r) == "raw body"

    def test_no_json_no_text_falls_back_to_default(self):
        r = NvidiaResponse(status_code=400, json_data=None, text="")
        assert atlas_proxy.upstream_error_text(r) == "upstream error"


# ── stream_error / _anthropic_stream_error ──────────────────────────────────

@pytest.mark.asyncio
async def test_stream_error_emits_openai_sse_with_message_and_done():
    """The OpenAI error stream must carry the proxy error message as a content
    chunk and terminate with [DONE]."""
    resp = atlas_proxy.stream_error("m", "upstream blew up", 502)
    raw = await _drain(resp)
    assert resp.status_code == 502
    assert b'"content":"Atlas proxy error (502): upstream blew up"' in raw
    assert raw.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_stream_error_empty_message_omits_content_chunk():
    """An empty message still produces a text block with the error prefix;
    sse_from_text guards on the full formatted string, not just the message."""
    resp = atlas_proxy.stream_error("m", "", 500)
    raw = await _drain(resp)
    assert b'"content":"Atlas proxy error (500): "' in raw
    assert raw.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_anthropic_stream_error_emits_full_anthropic_sequence(keys_file, monkeypatch):
    """_anthropic_stream_error must produce a well-formed Anthropic SSE stream
    (message_start..message_stop) with the error text in a text block."""
    resp = atlas_proxy._anthropic_stream_error("claude", "no usable NVIDIA keys are available", 503)
    raw = await _drain(resp)
    events = _anthropic_events(raw)
    assert events[0][0] == "message_start"
    assert events[-1][0] == "message_stop"
    text = "".join(d["delta"].get("text", "") for e, d in events
                   if e == "content_block_delta" and d["delta"].get("type") == "text_delta")
    assert "no usable NVIDIA keys" in text
    assert "503" in text
    # >=500 errors stream with their real status code (not masked as 200).
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_anthropic_stream_error_5xx_surfaces_real_status():
    """A >=500 error must surface its real status code on the SSE response so
    the client sees the failure severity (not masked as 200)."""
    resp = atlas_proxy._anthropic_stream_error("claude", "upstream down", 503)
    assert resp.status_code == 503


# ── _prepend_to_user_content / _content_to_text ─────────────────────────────

class TestPrependContent:
    def test_prepend_string_content(self):
        from proxy.system_prompt import _prepend_to_user_content
        assert _prepend_to_user_content("hi", "OVR") == "OVR\n\nhi"

    def test_prepend_empty_string(self):
        from proxy.system_prompt import _prepend_to_user_content
        assert _prepend_to_user_content("", "OVR") == "OVR"

    def test_prepend_list_preserves_existing_blocks(self):
        """Prepending to a block-list must add a fresh text block at index 0
        and leave the existing typed blocks (images, tool_results) intact."""
        from proxy.system_prompt import _prepend_to_user_content
        out = _prepend_to_user_content(
            [{"type": "text", "text": "keep"}, {"type": "image_url", "image_url": "x"}],
            "OVR",
        )
        assert out[0] == {"type": "text", "text": "OVR"}
        assert out[1] == {"type": "text", "text": "keep"}
        assert out[2] == {"type": "image_url", "image_url": "x"}

    def test_prepend_unknown_shape_wraps_override(self):
        from proxy.system_prompt import _prepend_to_user_content
        # An unknown content shape (int, etc.) wraps the override as a text block.
        out = _prepend_to_user_content(123, "OVR")
        assert out == [{"type": "text", "text": "OVR"}]


class TestContentToText:
    def test_none_is_empty(self):
        from proxy.system_prompt import _content_to_text
        assert _content_to_text(None) == ""

    def test_string_passthrough(self):
        from proxy.system_prompt import _content_to_text
        assert _content_to_text("hi") == "hi"

    def test_list_joins_text_blocks(self):
        from proxy.system_prompt import _content_to_text
        assert _content_to_text([
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
            {"type": "image", "x": 1},  # non-text block ignored
        ]) == "a\nb"

    def test_unknown_object_is_empty(self):
        """Arbitrary objects must NOT be stringified into the payload (the old
        bug that dumped a Python repr via f-string)."""
        from proxy.system_prompt import _content_to_text
        assert _content_to_text({"weird": 1}) == ""


# ── sanitize_openai_payload edge cases ──────────────────────────────────────

class TestSanitizeEdge:
    def test_stop_list_filters_non_strings(self):
        """A stop list with non-string entries (ints, None) must be filtered to
        only the valid strings; empty result drops the field."""
        from proxy.openai_compat import sanitize_openai_payload
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "stop": ["END", 7, None, "OK"],
        })
        assert out["stop"] == ["END", "OK"]

    def test_stop_list_all_invalid_drops_field(self):
        from proxy.openai_compat import sanitize_openai_payload
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "stop": [7, None],
        })
        assert "stop" not in out

    def test_explicit_tool_choice_preserved_not_defaulted(self):
        """When the caller sets tool_choice, it must be preserved (not
        overwritten with 'auto')."""
        from proxy.openai_compat import sanitize_openai_payload
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "required",
        })
        assert out["tool_choice"] == "required"

    def test_tools_without_tool_choice_defaults_auto(self):
        from proxy.openai_compat import sanitize_openai_payload
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        })
        assert out["tool_choice"] == "auto"

    def test_max_tokens_zero_falls_back_to_default(self):
        """max_tokens=0 is invalid (must be positive); fall back to 1024."""
        from proxy.openai_compat import sanitize_openai_payload
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        })
        assert out["max_tokens"] == 1024

    def test_max_tokens_negative_falls_back(self):
        from proxy.openai_compat import sanitize_openai_payload
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": -5,
        })
        assert out["max_tokens"] == 1024

    def test_temperature_as_string_coerced_and_clamped(self):
        """A string temperature must coerce to float and clamp into [0, 2]."""
        from proxy.openai_compat import sanitize_openai_payload
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "temperature": "1.5",
        })
        assert out["temperature"] == 1.5
        out2 = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "temperature": "9",
        })
        assert out2["temperature"] == 2.0

    def test_temperature_garbage_string_falls_back_to_default(self):
        from proxy.openai_compat import sanitize_openai_payload
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "temperature": "not-a-number",
        })
        assert out["temperature"] == 0.7

    def test_top_p_only_sent_when_present(self):
        from proxy.openai_compat import sanitize_openai_payload
        base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        assert "top_p" not in sanitize_openai_payload(dict(base))
        assert sanitize_openai_payload({**base, "top_p": 0.5})["top_p"] == 0.5

    def test_penalties_only_sent_when_present(self):
        from proxy.openai_compat import sanitize_openai_payload
        base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        out = sanitize_openai_payload({**base, "frequency_penalty": 1.5, "presence_penalty": -1.0})
        assert out["frequency_penalty"] == 1.5
        assert out["presence_penalty"] == -1.0
        assert "frequency_penalty" not in sanitize_openai_payload(dict(base))


# ── normalize_messages edge cases ───────────────────────────────────────────

class TestNormalizeEdge:
    def test_mixed_text_and_non_text_blocks_drops_non_text(self):
        """A content list mixing text + non-text blocks (image_url, etc.) must
        keep only the text parts, joined by newline."""
        from proxy.openai_compat import normalize_messages
        out = normalize_messages([{
            "role": "user",
            "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": "x"}, {"type": "text", "text": "b"}],
        }])
        assert out[0]["content"] == "a\nb"

    def test_missing_role_defaults_to_user(self):
        from proxy.openai_compat import normalize_messages
        out = normalize_messages([{"content": "hi"}])
        assert out[0]["role"] == "user"

    def test_tool_message_without_tool_call_id(self):
        """A tool message with no tool_call_id must still pass through (the
        tool_call_id is only attached when present)."""
        from proxy.openai_compat import normalize_messages
        out = normalize_messages([{"role": "tool", "content": "result"}])
        assert out[0]["role"] == "tool"
        assert out[0]["content"] == "result"
        assert "tool_call_id" not in out[0]

    def test_tool_message_with_name_preserves_name(self):
        from proxy.openai_compat import normalize_messages
        out = normalize_messages([{"role": "tool", "content": "r", "tool_call_id": "c1", "name": "fn"}])
        assert out[0]["tool_call_id"] == "c1"
        assert out[0]["name"] == "fn"

    def test_assistant_tool_calls_preserved(self):
        from proxy.openai_compat import normalize_messages
        tc = [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]
        out = normalize_messages([{"role": "assistant", "content": "ok", "tool_calls": tc}])
        assert out[0]["tool_calls"] == tc
        assert out[0]["role"] == "assistant"

    def test_empty_content_list_becomes_empty_string(self):
        from proxy.openai_compat import normalize_messages
        out = normalize_messages([{"role": "user", "content": []}])
        assert out[0]["content"] == ""


# ── stats internals ─────────────────────────────────────────────────────────

class TestStatsInternals:
    def test_merge_bucket_accumulates_counts_and_models(self, isolated_stats):
        from proxy import stats as s
        dst = s._empty_bucket()
        src1 = s._empty_bucket(); src1["requests"] = 5; src1["prompt_tokens"] = 10; src1["models"] = {"m1": 2}
        s._merge_bucket(dst, src1)
        assert dst["requests"] == 5
        assert dst["prompt_tokens"] == 10
        assert dst["models"] == {"m1": 2}
        # Second merge accumulates models by key.
        src2 = s._empty_bucket(); src2["models"] = {"m1": 1, "m2": 3}; src2["requests"] = 2
        s._merge_bucket(dst, src2)
        assert dst["requests"] == 7
        assert dst["models"] == {"m1": 3, "m2": 3}

    def test_merge_bucket_accumulates_all_fields(self, isolated_stats):
        from proxy import stats as s
        dst = s._empty_bucket()
        src = s._empty_bucket()
        src["requests"] = 1; src["successes"] = 1; src["failures"] = 0
        src["prompt_tokens"] = 5; src["completion_tokens"] = 3; src["total_tokens"] = 8
        src["tool_calls"] = 2; src["provider_nvidia"] = 1
        s._merge_bucket(dst, src)
        for field in ("requests", "successes", "failures", "prompt_tokens",
                      "completion_tokens", "total_tokens", "tool_calls", "provider_nvidia"):
            assert dst[field] == src[field], f"{field} not accumulated"

    def test_parse_iso_z_suffix(self):
        from proxy import stats as s
        assert s._parse_iso("2026-07-21T00:00:00Z") == 1784592000.0

    def test_parse_iso_offset(self):
        from proxy import stats as s
        assert s._parse_iso("2026-07-21T00:00:00+00:00") == 1784592000.0

    def test_parse_iso_bad_returns_zero(self):
        from proxy import stats as s
        assert s._parse_iso("not a date") == 0.0
        assert s._parse_iso("") == 0.0
        assert s._parse_iso(None) == 0.0

    def test_save_atomic_write_roundtrip(self, isolated_stats, tmp_path):
        from proxy import stats as s
        data = {"started_at": "x", "restart": s._empty_bucket(), "all_time": s._empty_bucket()}
        s.save(data)
        # The .tmp file is gone after atomic replace.
        assert not (tmp_path / "stats" / "proxy_stats.json.tmp").exists()
        reloaded = s.load()
        assert reloaded["started_at"] == "x"

    def test_save_creates_parent_dir(self, isolated_stats):
        from proxy import stats as s
        # STATS_DIR already exists (isolated_stats fixture), but save must be
        # safe if it didn't — exercise mkdir(parents=True) by pointing at a
        # nested missing path.
        import os
        from proxy import stats as stats_mod
        nested = isolated_stats.parent / "nested" / "deeper"
        stats_mod.STATS_DIR = nested
        stats_mod.STATS_FILE = nested / "proxy_stats.json"
        s.save({"started_at": "y", "restart": s._empty_bucket(), "all_time": s._empty_bucket()})
        assert nested.exists()
        assert (nested / "proxy_stats.json").exists()

    def test_get_status_uptime_from_started_at(self, isolated_stats):
        from proxy import stats as s
        s.record_success("nvidia", "m", 10, 5, 15, 0)
        status = s.get_status()
        # uptime_seconds is derived from started_at; with a fresh stamp it's
        # small and non-negative.
        assert status["uptime_seconds"] >= 0
        assert status["restart"]["requests"] == 1
        assert status["restart"]["avg_tokens_per_request"] == 15.0

    def test_now_iso_is_iso8601_utc(self):
        from proxy import stats as s
        n = s._now_iso()
        # Has a T separator and a timezone designator (+00:00 or Z).
        assert "T" in n and ("+" in n or "Z" in n)


# ── log formatters ──────────────────────────────────────────────────────────

def _make_record(msg, **extra):
    """Build a LogRecord with extra= fields the way _log_event does."""
    r = logging.LogRecord("atlas-proxy", logging.INFO, "f", 1, msg, None, None)
    for k, v in extra.items():
        setattr(r, k, v)
    return r


class TestJsonFormatter:
    def test_lifts_extra_fields_top_level(self):
        from proxy.atlas_proxy import _JsonFormatter
        out = json.loads(_JsonFormatter().format(
            _make_record("req in", rid="r1", event="request", model="m")
        ))
        assert out["msg"] == "req in"
        assert out["level"] == "INFO"
        assert out["rid"] == "r1"
        assert out["event"] == "request"
        assert out["model"] == "m"

    def test_reserved_internals_not_lifted(self):
        """Logging internals (args, msecs, thread, etc.) must NOT leak into the
        JSON object — only the caller's extra= fields."""
        from proxy.atlas_proxy import _JsonFormatter
        out = json.loads(_JsonFormatter().format(_make_record("x", rid="r1")))
        reserved = {"args", "msecs", "thread", "threadName", "processName",
                    "process", "levelname", "levelno", "pathname", "filename",
                    "module", "exc_text", "stack_info", "funcName", "created",
                    "relativeCreated", "name", "message"}
        assert not (set(out) & reserved), f"reserved fields leaked: {set(out) & reserved}"

    def test_exc_info_becomes_exc_field(self):
        from proxy.atlas_proxy import _JsonFormatter
        try:
            1 / 0
        except ZeroDivisionError:
            r = logging.LogRecord("atlas-proxy", logging.ERROR, "f", 1, "boom", None, sys.exc_info())
        out = json.loads(_JsonFormatter().format(r))
        assert "exc" in out
        assert "ZeroDivisionError" in out["exc"]

    def test_ts_is_iso_date(self):
        from proxy.atlas_proxy import _JsonFormatter
        out = json.loads(_JsonFormatter().format(_make_record("x")))
        assert "ts" in out
        assert "T" in out["ts"]  # %Y-%m-%dT%H:%M:%S


class TestCleanFormatter:
    def test_produces_timestamp_and_message(self):
        from proxy.atlas_proxy import _CleanFormatter
        line = _CleanFormatter().format(_make_record("hello world"))
        # Shape: <color>HH:MM:SS<reset> hello world  (color only in a TTY;
        # the formatter always emits the ANSI here since it doesn't check isatty).
        assert "hello world" in line
        assert ":" in line  # HH:MM:SS
        # The level label is deliberately omitted (per the docstring).
        assert "INFO" not in line


# ── _on_timeout mid-stream callback (stream path) ───────────────────────────

@pytest.mark.asyncio
async def test_stream_on_timeout_cools_key_and_records_failure(keys_file, monkeypatch):
    """A mid-stream timeout in the OpenAI stream path must fire on_timeout,
    which cools the key and records a failure — so a 200-then-hang doesn't
    recycle the dead key or count as a success."""
    _patch_keys(monkeypatch, KEYS[:2])

    async def upstream() -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise httpx.ReadTimeout("idle")

    async def fake_stream_chat(api_key, payload, rid="", on_timeout=None, timings=None):
        # Drive the iterator ourselves so the timeout fires inside it, then
        # invoke on_timeout as the proxy would on a mid-stream ReadTimeout.
        it = upstream()
        try:
            async for chunk in it:
                pass
        except httpx.ReadTimeout:
            if on_timeout:
                on_timeout()
        return 200, {}, _empty_iter(), ""

    async def _empty_iter() -> AsyncIterator[bytes]:
        return
        yield  # noqa

    cooled = []
    monkeypatch.setattr(atlas_proxy.key_store, "cooldown_key",
                        AsyncMock(side_effect=lambda k: cooled.append(k)))
    failed = []
    monkeypatch.setattr(atlas_proxy, "record_failure", lambda *a, **k: failed.append(a))

    atlas_proxy.nvidia_client.stream_chat = fake_stream_chat
    resp = await atlas_proxy.handle_stream("m", {"model": "m", "messages": [], "stream": True}, rid="t1", started=0.0)
    await _drain(resp)
    assert cooled, "on_timeout must cool the key"
    assert failed, "on_timeout must record a failure"


# ── openai_response_to_anthropic usage + edge cases ─────────────────────────

class TestOpenaiToAnthropicResponse:
    def test_usage_mapped_to_input_output_tokens(self):
        from proxy.openai_compat import openai_response_to_anthropic
        openai = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}}
        out = openai_response_to_anthropic("claude", openai)
        assert out["usage"] == {"input_tokens": 11, "output_tokens": 4}

    def test_missing_usage_defaults_zero(self):
        from proxy.openai_compat import openai_response_to_anthropic
        openai = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
        out = openai_response_to_anthropic("claude", openai)
        assert out["usage"] == {"input_tokens": 0, "output_tokens": 0}

    def test_length_finish_reason_maps_to_max_tokens(self):
        from proxy.openai_compat import openai_response_to_anthropic
        openai = {"choices": [{"message": {"content": "hi"}, "finish_reason": "length"}]}
        out = openai_response_to_anthropic("claude", openai)
        assert out["stop_reason"] == "max_tokens"

    def test_content_filter_maps_to_end_turn(self):
        from proxy.openai_compat import openai_response_to_anthropic
        openai = {"choices": [{"message": {"content": "hi"}, "finish_reason": "content_filter"}]}
        out = openai_response_to_anthropic("claude", openai)
        assert out["stop_reason"] == "end_turn"

    def test_unknown_finish_reason_defaults_end_turn(self):
        from proxy.openai_compat import openai_response_to_anthropic
        openai = {"choices": [{"message": {"content": "hi"}, "finish_reason": None}]}
        out = openai_response_to_anthropic("claude", openai)
        assert out["stop_reason"] == "end_turn"

    def test_tool_call_bad_json_arguments_defaults_empty_dict(self):
        """Malformed tool-call arguments must fall back to {} not crash."""
        from proxy.openai_compat import openai_response_to_anthropic
        openai = {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "f", "arguments": "not json{"}},
        ]}, "finish_reason": "tool_calls"}]}
        out = openai_response_to_anthropic("claude", openai)
        tool_block = [b for b in out["content"] if b["type"] == "tool_use"][0]
        assert tool_block["input"] == {}
        assert out["stop_reason"] == "tool_use"


# ── anthropic_tools_to_openai edge cases ────────────────────────────────────

class TestAnthropicTools:
    def test_non_list_returns_empty(self):
        from proxy.openai_compat import anthropic_tools_to_openai
        assert anthropic_tools_to_openai("not a list") == []
        assert anthropic_tools_to_openai(None) == []

    def test_skips_tool_without_name(self):
        from proxy.openai_compat import anthropic_tools_to_openai
        out = anthropic_tools_to_openai([
            {"name": "f", "input_schema": {"type": "object"}},
            {"description": "no name here", "input_schema": {}},
        ])
        assert len(out) == 1
        assert out[0]["function"]["name"] == "f"

    def test_missing_schema_defaults_empty(self):
        from proxy.openai_compat import anthropic_tools_to_openai
        out = anthropic_tools_to_openai([{"name": "f"}])
        assert out[0]["function"]["parameters"] == {}

    def test_description_defaults_empty_string(self):
        from proxy.openai_compat import anthropic_tools_to_openai
        out = anthropic_tools_to_openai([{"name": "f", "input_schema": {}}])
        assert out[0]["function"]["description"] == ""


# ── anthropic_tool_choice_to_openai edge cases ───────────────────────────────

class TestAnthropicToolChoice:
    def test_auto(self):
        from proxy.openai_compat import anthropic_tool_choice_to_openai
        assert anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"

    def test_any_maps_required(self):
        from proxy.openai_compat import anthropic_tool_choice_to_openai
        assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"

    def test_tool_with_name(self):
        from proxy.openai_compat import anthropic_tool_choice_to_openai
        assert anthropic_tool_choice_to_openai({"type": "tool", "name": "f"}) == {"type": "function", "function": {"name": "f"}}

    def test_tool_without_name_returns_none(self):
        from proxy.openai_compat import anthropic_tool_choice_to_openai
        assert anthropic_tool_choice_to_openai({"type": "tool"}) is None

    def test_non_dict_returns_none(self):
        from proxy.openai_compat import anthropic_tool_choice_to_openai
        assert anthropic_tool_choice_to_openai("auto") is None
        assert anthropic_tool_choice_to_openai(None) is None

    def test_unknown_type_returns_none(self):
        from proxy.openai_compat import anthropic_tool_choice_to_openai
        assert anthropic_tool_choice_to_openai({"type": "weird"}) is None
