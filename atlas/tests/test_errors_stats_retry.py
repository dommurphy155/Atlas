"""nvidia_client error extraction + stats recording + retry/failover.

Error extraction decides what message the client sees on a 400; stats decides
what the operator sees in `atlas tokens`; the failover loop decides which key
serves the next request after a failure. All three are high-regression-risk
and all three are fully testable without the NVIDIA API.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from proxy.nvidia_client import _extract_error_message, _timeout_kind
from proxy import stats as stats_mod


# ── error extraction ─────────────────────────────────────────────────────────

class TestExtractError:
    def test_nvidia_native_message(self):
        body = json.dumps({"message": "Validation: Temperature must be between 0 and 2, got 2.5", "type": "Bad Request", "code": 400})
        assert _extract_error_message(body) == "Validation: Temperature must be between 0 and 2, got 2.5"

    def test_openai_error_shape(self):
        body = json.dumps({"error": {"message": "rate limited", "type": "x", "code": 429}})
        assert _extract_error_message(body) == "rate limited"

    def test_fastapi_detail_shape(self):
        body = json.dumps({"detail": "missing field foo"})
        assert _extract_error_message(body) == "missing field foo"

    def test_nvidia_429_title_shape(self):
        body = json.dumps({"status": 429, "title": "Too Many Requests"})
        assert _extract_error_message(body) == "Too Many Requests"

    def test_empty_body(self):
        assert _extract_error_message("") == "upstream error"

    def test_non_json_body_returns_raw(self):
        assert _extract_error_message("plain text error") == "plain text error"

    def test_truncates_long_raw(self):
        long = "x" * 1000
        assert len(_extract_error_message(long)) <= 500


def test_timeout_kind_classification():
    import httpx
    assert _timeout_kind(httpx.ConnectTimeout("x")) == "connect_timeout"
    assert _timeout_kind(httpx.ReadTimeout("x")) == "idle_read"
    assert _timeout_kind(httpx.PoolTimeout("x")) == "pool_timeout"
    assert _timeout_kind(httpx.WriteTimeout("x")) == "write_timeout"
    assert _timeout_kind(httpx.TimeoutException("x")) == "timeout"


# ── stats ────────────────────────────────────────────────────────────────────

class TestStats:
    def test_record_success_updates_buckets(self, isolated_stats):
        stats_mod.record_success("nvidia", "z-ai/glm-5.2", 100, 50, 150, 2)
        status = stats_mod.get_status()
        assert status["restart"]["requests"] == 1
        assert status["restart"]["successes"] == 1
        assert status["restart"]["prompt_tokens"] == 100
        assert status["restart"]["completion_tokens"] == 50
        assert status["restart"]["tool_calls"] == 2
        assert status["all_time"]["requests"] == 1
        assert status["all_time"]["models"]["z-ai/glm-5.2"] == 1

    def test_record_failure_updates_failures(self, isolated_stats):
        stats_mod.record_failure("nvidia")
        status = stats_mod.get_status()
        assert status["restart"]["failures"] == 1
        assert status["all_time"]["failures"] == 1

    def test_reset_since_restart_clears_restart_only(self, isolated_stats):
        stats_mod.record_success("nvidia", "m", 10, 5, 15, 0)
        stats_mod.reset_since_restart()
        status = stats_mod.get_status()
        assert status["restart"]["requests"] == 0
        assert status["all_time"]["requests"] == 1, "all_time must survive a restart reset"

    def test_persistence_roundtrip(self, isolated_stats):
        stats_mod.record_success("nvidia", "m", 1, 1, 2, 0)
        # Re-load from disk (simulating a restart).
        data = stats_mod.load()
        assert data["restart"]["requests"] == 1
        assert data["all_time"]["requests"] == 1

    def test_corrupt_file_recovers_to_empty(self, isolated_stats):
        isolated_stats.write_text("not json at all {{{")
        data = stats_mod.load()
        assert data["restart"]["requests"] == 0
        assert "all_time" in data


# ── retry / failover loop ────────────────────────────────────────────────────
#
# The failover loop lives in atlas_proxy._stream_failover_loop. We exercise it
# via handle_non_stream with a mocked NvidiaClient so we can feed canned
# NVIDIA responses (429, 5xx, 200) and assert the key-rotation + retry
# behavior without touching the network.

@pytest.mark.asyncio
async def test_non_stream_failover_on_429_rotates_key(keys_file, monkeypatch):
    """A 429 cools the key and rotates to the next; the request succeeds on key #2."""
    from proxy import atlas_proxy
    from proxy.nvidia_client import NvidiaResponse

    keys = [f"nvapi-k{i:02d}" + "x" * 40 for i in range(3)]
    keys_file.write_text("\n".join(keys) + "\n")
    monkeypatch.setattr(atlas_proxy.key_store, "_keys", keys)
    monkeypatch.setattr(atlas_proxy.key_store, "_active_index", -1)

    responses = [
        NvidiaResponse(status_code=429, json_data={"error": {"message": "rate limited"}}),
        NvidiaResponse(status_code=200, json_data={
            "id": "x", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }),
    ]
    atlas_proxy.nvidia_client.chat = AsyncMock(side_effect=responses)

    result = await atlas_proxy.handle_non_stream("z-ai/glm-5.2", {"model": "z-ai/glm-5.2", "messages": []}, rid="t1", started=0.0)
    assert result.status_code == 200
    body = json.loads(result.body.decode())
    assert body["choices"][0]["message"]["content"] == "ok"
    # Called twice: once (429), once (200 on the rotated key).
    assert atlas_proxy.nvidia_client.chat.await_count == 2


@pytest.mark.asyncio
async def test_non_stream_retries_5xx_up_to_max(keys_file, monkeypatch):
    """Transient 5xx retries on the same pool up to MAX_RETRIES, then surfaces."""
    from proxy import atlas_proxy
    from proxy.nvidia_client import NvidiaResponse

    keys = [f"nvapi-k{i:02d}" + "x" * 40 for i in range(3)]
    keys_file.write_text("\n".join(keys) + "\n")
    monkeypatch.setattr(atlas_proxy.key_store, "_keys", keys)
    monkeypatch.setattr(atlas_proxy.key_store, "_active_index", -1)
    monkeypatch.setattr(atlas_proxy, "MAX_RETRIES", 2)

    # Always 503 → retries MAX_RETRIES then surfaces 503.
    atlas_proxy.nvidia_client.chat = AsyncMock(return_value=NvidiaResponse(status_code=503, json_data={"error": {"message": "upstream down"}}))

    result = await atlas_proxy.handle_non_stream("m", {"model": "m", "messages": []}, rid="t2", started=0.0)
    assert result.status_code == 503
    # Initial attempt + MAX_RETRIES retries = 3 calls total.
    assert atlas_proxy.nvidia_client.chat.await_count == 3


@pytest.mark.asyncio
async def test_non_stream_400_surfaces_real_message(keys_file, monkeypatch):
    """A non-failover 4xx must surface the real upstream message, not a bare status."""
    from proxy import atlas_proxy
    from proxy.nvidia_client import NvidiaResponse

    keys = [f"nvapi-k{i:02d}" + "x" * 40 for i in range(2)]
    keys_file.write_text("\n".join(keys) + "\n")
    monkeypatch.setattr(atlas_proxy.key_store, "_keys", keys)
    monkeypatch.setattr(atlas_proxy.key_store, "_active_index", -1)

    atlas_proxy.nvidia_client.chat = AsyncMock(return_value=NvidiaResponse(
        status_code=400, json_data={"error": {"message": "Validation: bad tool schema"}}))

    result = await atlas_proxy.handle_non_stream("m", {"model": "m", "messages": []}, rid="t3", started=0.0)
    assert result.status_code == 400
    body = json.loads(result.body.decode())
    assert "Validation: bad tool schema" in body["error"]["message"]


@pytest.mark.asyncio
async def test_non_stream_no_keys_returns_503(monkeypatch):
    """An empty key pool must 503 with 'no usable NVIDIA keys' rather than loop."""
    from proxy import atlas_proxy

    monkeypatch.setattr(atlas_proxy.key_store, "_keys", [])
    monkeypatch.setattr(atlas_proxy.key_store, "_active_index", -1)
    atlas_proxy.nvidia_client.chat = AsyncMock()

    result = await atlas_proxy.handle_non_stream("m", {"model": "m", "messages": []}, rid="t4", started=0.0)
    assert result.status_code == 503
    assert atlas_proxy.nvidia_client.chat.await_count == 0, "must not call NVIDIA with no keys"
