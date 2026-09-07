"""Response-shape conversion: OpenAI <-> Anthropic.

Two finish_reason mappers (intentionally different -- the public one
covers the full set of well-known values; the SSE-internal one is a
tighter subset for the streaming path) plus usage-object translation
and the non-streaming response shape converter.

``promote_reasoning_to_content_in_chat_response`` lives here too --
it's the OpenAI-side counterpart of the reasoning-content fallback
that ``openai_response_to_anthropic`` already does for /v1/messages.
Restored 2026-09-06 per the user's standing rule on stripped features.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from ..config import get_default_model


# ---------------------------------------------------------------------------
# Finish reason & usage (response path)
# ---------------------------------------------------------------------------

def map_finish_reason_openai_to_anthropic(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "function_call": "tool_use",
    }.get(reason, reason)


def map_finish_reason_anthropic_to_openai(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "pause_turn": "stop",
        "refusal": "content_filter",
    }.get(reason, reason)


def translate_usage_anthropic_to_openai(usage: Optional[Dict]) -> Optional[Dict]:
    if not usage:
        return usage
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": (
            usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        ),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens")
        or ((usage.get("output_tokens_details") or {}).get("reasoning_tokens")),
    }


def translate_usage_openai_to_anthropic(usage: Optional[Dict]) -> Optional[Dict]:
    if not usage:
        return usage
    out: Dict[str, Any] = {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "output_tokens": usage.get(
            "completion_tokens", usage.get("output_tokens", 0)
        ),
    }
    details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )
    if "reasoning_tokens" in details or "reasoning_tokens" in usage:
        out["output_tokens_details"] = {
            "reasoning_tokens": details.get("reasoning_tokens")
            or usage.get("reasoning_tokens"),
        }
    for k in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        if k in usage:
            out[k] = usage[k]
    return out


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic non-streaming response translation
# ---------------------------------------------------------------------------

# Internal finish_reason mapper for the SSE/response path. Different from
# the public map_finish_reason_openai_to_anthropic above -- this one always
# returns a non-None value (defaults to "end_turn") because the
# Anthropic-shaped response is required to have stop_reason.
def _map_openai_finish_reason_to_anthropic(reason: Optional[str]) -> str:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_calls",
        "abort": "end_turn",
        "content_filter": "stop",
    }
    return mapping.get(reason or "", "end_turn")


def openai_response_to_anthropic(data: Dict[str, Any], *, rid: str = "") -> Dict[str, Any]:
    """Convert a non-streaming OpenAI chat/completions response to the
    Anthropic /messages response shape.

    Input (OpenAI-style):
        {"id": "...", "choices": [{"message": {"role":"assistant","content":"...","tool_calls":[...]}}], "usage": {...}}

    Output (Anthropic-style):
        {"id": "...", "type": "message", "role": "assistant", "content": [...], "usage": {...}, ...}
    """
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content_str = msg.get("content") or ""
    reasoning_str = msg.get("reasoning_content") or ""
    # Some upstream models (notably NVIDIA reasoning models) emit
    # content: None and put the actual answer in reasoning_content. The
    # Anthropic SDK expects a non-empty content array, so fall back.
    if not content_str and reasoning_str:
        content_str = reasoning_str
    tool_calls = msg.get("tool_calls") or []

    content_blocks: List[Dict[str, Any]] = []
    if isinstance(content_str, str) and content_str:
        content_blocks.append({"type": "text", "text": content_str})

    tool_blocks: List[Dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        tool_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
            "name": fn.get("name", ""),
            "input": fn.get("arguments", ""),
        })

    out: Dict[str, Any] = {
        "id": data.get("id") or rid or f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": get_default_model(),
        "content": content_blocks + tool_blocks,
        "stop_reason": _map_openai_finish_reason_to_anthropic(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": translate_usage_openai_to_anthropic(data.get("usage")),
    }
    return out


# ---------------------------------------------------------------------------
# OpenAI -> OpenAI reasoning-content promotion
# ---------------------------------------------------------------------------

def promote_reasoning_to_content_in_chat_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """For OpenAI chat/completions responses where the model emitted
    ``content: null`` and put the actual answer in ``reasoning_content``,
    promote ``reasoning_content`` -> ``content`` so OpenAI clients (which
    ignore ``reasoning_content``) see a non-empty response instead of an
    apparently-hung model.

    Walks every choice's message in-place; returns the same dict for
    chaining. This is the OpenAI-side counterpart of the Anthropic-side
    fallback in ``openai_response_to_anthropic`` (which already does this
    for /v1/messages).

    Per the atlas-proxy-codebase skill: this function was stripped
    2026-09-02 during the Codex/Responses-API stripdown. It is restored
    here per the user's standing rule ("when a feature is stripped,
    restore it at the original location with original shape").
    """
    if not isinstance(data, dict):
        return data
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        reasoning = msg.get("reasoning_content")
        # Only promote when content is truly empty/null AND reasoning has
        # something to show. If content already has text, leave it alone.
        if (not content) and isinstance(reasoning, str) and reasoning:
            msg["content"] = reasoning
    return data
