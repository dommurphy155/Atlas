"""Security/observability hardening tests for atlas proxy fork."""
import os
import stat
import tempfile
import pytest
from pathlib import Path


def test_redact_frame_for_log_caps_at_max_bytes() -> None:
    """SSE error frames must be hard-capped at MAX_FRAME_LOG_BYTES."""
    from proxy.proxy import _redact_frame_for_log, MAX_FRAME_LOG_BYTES

    big = b'data: {"error": {"message": "' + b"x" * 10000 + b'"}}'
    out = _redact_frame_for_log(big)
    assert len(out) <= MAX_FRAME_LOG_BYTES, f"len={len(out)} > MAX={MAX_FRAME_LOG_BYTES}"


def test_redact_frame_for_log_strips_sensitive_values() -> None:
    """Sensitive JSON keys (prompt, api_key, etc.) must have their values redacted."""
    from proxy.proxy import _redact_frame_for_log

    frame = b'data: {"prompt": "secret user prompt here", "error": "rate limited"}'
    out = _redact_frame_for_log(frame)
    assert "secret user prompt here" not in out, f"prompt value leaked: {out}"
    assert "[REDACTED]" in out
    # The error key should not be redacted (not in our sensitive list)
    assert "rate limited" in out


def test_redact_frame_for_log_strips_authorization() -> None:
    """Authorization header values in error payloads must be redacted."""
    from proxy.proxy import _redact_frame_for_log

    frame = b'data: {"Authorization": "Bearer sk-very-secret", "msg": "fail"}'
    out = _redact_frame_for_log(frame)
    assert "sk-very-secret" not in out
    assert "[REDACTED]" in out
    assert "fail" in out


def test_redact_frame_for_log_handles_non_utf8() -> None:
    """Non-UTF-8 bytes should not crash; use errors=replace."""
    from proxy.proxy import _redact_frame_for_log

    frame = b"data: \xff\xfe invalid utf8"
    out = _redact_frame_for_log(frame)
    # Should return a string without raising
    assert isinstance(out, str)


def test_retry_statuses_excludes_409() -> None:
    """R#11: 409 (Conflict) is not safe to auto-retry — must NOT be in RETRY_STATUSES."""
    from proxy.config import RETRY_STATUSES
    assert 409 not in RETRY_STATUSES, "409 must not be retried automatically"


def test_world_writable_runtime_provider_rejected(tmp_path, monkeypatch) -> None:
    """S#6: world-writable runtime_provider.json must be rejected."""
    from proxy import config

    # Create a world-writable runtime provider file
    runtime = tmp_path / "runtime_provider.json"
    runtime.write_text('{"provider": "huggingface"}')
    os.chmod(runtime, 0o666)  # world-writable

    # Point the config module at it
    monkeypatch.setattr(config, "RUNTIME_PROVIDER_FILE", str(runtime))
    monkeypatch.setenv("ATLAS_PROVIDER", "openrouter")

    # The loader must fall back to env var, NOT pick "huggingface"
    result = config._load_runtime_provider()
    assert result == "openrouter", f"loader trusted world-writable file: got {result}"


def test_owned_runtime_provider_accepted(tmp_path, monkeypatch) -> None:
    """A normal (non-world-writable) runtime_provider.json is loaded."""
    from proxy import config

    runtime = tmp_path / "runtime_provider.json"
    runtime.write_text('{"provider": "huggingface"}')
    # Default file perms (0o644) are not world-writable
    os.chmod(runtime, 0o644)

    monkeypatch.setattr(config, "RUNTIME_PROVIDER_FILE", str(runtime))
    result = config._load_runtime_provider()
    assert result == "huggingface"
