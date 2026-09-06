"""SSE frame classification, redaction, and lifecycle helpers.

This module centralises everything that inspects, classifies, or redacts a
single SSE frame. It is provider-agnostic — the rules it enforces (HF
rate-limit, OpenAI error shape, Anthropic error shape) are encoded once
and re-used by every upstream-stream iterator in the proxy.

Extracted from `proxy.py` (Round 3 architecture review, finding #1) so
the streaming path's two near-identical generators can share these
helpers without duplicating them.
"""
from __future__ import annotations

import json as _json
import re as _re
from typing import Any, Dict, Optional

# --- atlas-sse-frame-redact-helper v1 ---
# Cap and redact an SSE error frame for safe logging. The frame is JSON
# (OpenAI / Anthropic / HF error shapes). We strip values of common
# sensitive keys ("prompt", "input", "messages", "system", "content",
# "Authorization", "api_key") and hard-cap the result at MAX_FRAME_LOG_BYTES
# to prevent log bombs.
_REDACT_KEYS = (
    "prompt", "input", "messages", "system", "content",
    "Authorization", "api_key", "apiKey", "x-api-key",
    "hf_token", "OPENAI_API_KEY",
)
_REDACT_RE = _re.compile(
    r'"(?:' + "|".join(_REDACT_KEYS) + r')"\s*:\s*"(?:[^"\\]|\\.)*"',
    _re.IGNORECASE,
)
MAX_FRAME_LOG_BYTES = 80


def redact_frame_for_log(frame: bytes) -> str:
    """Return a short, redacted string representation of a frame for logging.

    Hard-cap at MAX_FRAME_LOG_BYTES (80). Strips values of common sensitive
    keys so an upstream that echoes the user's prompt in an error payload
    does not leak it into the operator's logs.
    """
    try:
        s = frame.decode("utf-8", errors="replace")
    except Exception:
        s = repr(frame)
    s = _REDACT_RE.sub(lambda m: m.group(0).split(":", 1)[0] + ':"[REDACTED]"', s)
    return s[:MAX_FRAME_LOG_BYTES]


# --- atlas-sse-error-classification-patch v1 ---
def classify_sse_frame(frame: bytes) -> Optional[dict]:
    """
    Structurally classify a single SSE frame as either a provider error
    event, or ordinary content (return None).

    This function NEVER substring-matches the raw frame text. It only
    inspects:
      - the SSE `event:` field name (e.g. "error")
      - the JSON payload of the `data:` field, and only well-known error
        shape keys within it: top-level "type"=="error", a nested "error"
        object, or an OpenAI-style top-level "error" object.

    Ordinary content frames (content_block_delta, message text, tool-call
    argument deltas, etc.) do not have this shape and are always returned
    as None, regardless of what English words their content contains
    (e.g. "concurrently", "worker", "capacity", "rpm" in model output).

    Returns a dict {"kind": str, "raw_type": str|None, "message": str,
    "code": Any} for a genuine structured error frame, else None.
    """
    event_name = None
    data_raw = None
    for line in frame.split(b"\n"):
        line = line.strip(b"\r")
        if line.startswith(b"event:"):
            event_name = line[len(b"event:"):].strip().decode("utf-8", "ignore")
        elif line.startswith(b"data:"):
            piece = line[len(b"data:"):].strip()
            data_raw = piece if data_raw is None else data_raw + piece

    is_named_error_event = event_name == "error"

    obj = None
    if data_raw and data_raw != b"[DONE]":
        try:
            obj = _json.loads(data_raw)
        except Exception:
            obj = None

    if not isinstance(obj, dict):
        # A named `event: error` with a non-JSON/empty body is still
        # structurally an error event (rare, but don't silently swallow it).
        if is_named_error_event:
            return {"kind": "generic_error", "raw_type": None, "message": "", "code": None}
        return None

    top_type = obj.get("type")
    err_obj = obj.get("error") if isinstance(obj.get("error"), dict) else None

    # Structural gate: only proceed if this frame is actually shaped like an
    # error (named error event, top-level type=="error", or an "error" key).
    # A content_block_delta / text / tool-call-argument frame never matches
    # this shape, so it falls through untouched no matter its text content.
    if not is_named_error_event and top_type != "error" and err_obj is None:
        return None

    err_type = ""
    err_message = ""
    err_code = None
    if err_obj:
        err_type = str(err_obj.get("type") or "").lower()
        err_message = str(err_obj.get("message") or "")
        err_code = err_obj.get("code")
    elif top_type == "error":
        err_message = str(obj.get("message") or "")

    hay = (err_type + " " + err_message).lower()

    def _has(*words: str) -> bool:
        return any(w in hay for w in words)

    if _has(
        "context_length", "context length", "too many tokens",
        "maximum context", "prompt is too long", "input length",
        "token limit", "context_length_exceeded",
    ):
        kind = "context_length"
    elif _has("rate_limit", "rate limit", "tpm", "rpm", "too many requests"):
        kind = "rate_limit"
    elif _has("concurrent", "concurrency", "max_concurrent"):
        kind = "concurrency"
    elif _has("idle timeout", "upstream idle", "idle_timeout"):
        kind = "idle_timeout"
    elif _has(
        "overloaded", "resourceexhausted", "resource_exhausted",
        "capacity", "unavailable", "server_error",
    ):
        kind = "overloaded"
    else:
        kind = "generic_error"

    return {"kind": kind, "raw_type": err_type or top_type, "message": err_message, "code": err_code}


def is_content_sse_frame(frame: bytes) -> bool:
    """
    Structurally determine whether an SSE frame carries *visible* content
    for the Anthropic /messages protocol.

    Returns True for:
      - content_block_delta with delta.type == "text_delta" or
        delta.type == "input_json_delta" (visible assistant content)
      - content_block_start with any content_block.type that carries
        visible output (text, tool_use, etc.)
      - message_start (the assistant has begun emitting, so the stream
        produced something meaningful even if no delta arrived)
    """
    if b"data:" not in frame:
        return False
    data_raw = None
    for line in frame.split(b"\n"):
        line = line.strip(b"\r")
        if line.startswith(b"data:"):
            data_raw = line[5:].strip()
            break
    if not data_raw or data_raw == b"[DONE]":
        return False
    try:
        obj = _json.loads(data_raw)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    top_type = obj.get("type", "")
    if top_type == "content_block_delta":
        delta = obj.get("delta") or {}
        delta_type = delta.get("type", "")
        # text_delta carries visible text; input_json_delta carries
        # partial tool-call arguments; thinking_delta carries the model's
        # reasoning trace — all three are assistant-side content.
        if delta_type in ("text_delta", "input_json_delta", "thinking_delta"):
            return True
    if top_type == "content_block_start":
        block = obj.get("content_block") or {}
        block_type = block.get("type", "")
        # text and tool_use blocks both represent assistant output
        if block_type in ("text", "tool_use"):
            return True
    if top_type == "message_start":
        return True
    return False


# Map the structured error kind to a status code (used for mark_error and
# for selecting a retry decision later).
KIND_TO_STATUS: Dict[str, int] = {
    "rate_limit": 429,
    "concurrency": 429,
    "idle_timeout": 408,
    "overloaded": 503,
    "context_length": 400,
    "generic_error": 502,
}


def is_openai_done_frame(frame: bytes) -> bool:
    """Return True if the frame carries an OpenAI [DONE] sentinel, OR is an
    empty (keepalive) frame that should be treated as a done marker.

    The OpenAI chat-completions stream terminator is ``[DONE]``. Some
    Anthropic clients get confused by it; callers must gate the [DONE]
    emission on whether the client expects the OpenAI shape.

    Mirrors the original `proxy.utils.is_openai_done_frame` behaviour:
    an empty frame (after trimming CRLF) is also treated as a done
    marker because some upstreams send a bare newline to flush.
    """
    text = frame.replace(b"\r\n", b"\n").strip()
    if not text:
        return True
    for line in text.split(b"\n"):
        line = line.strip()
        if line.startswith(b"data:"):
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return True
        if line == b"event: data":
            if b"[DONE]" in text:
                return True
    return False
