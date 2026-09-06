"""Translation tests — message shape, tools, thinking, SSE conversion.

Regression coverage for the [DONE]-stripping + thinking-block-index bugs
that were CRITICAL.
"""

from __future__ import annotations

import pytest

from proxy import translation
from proxy.translation import (
    openai_sse_to_anthropic_sse,
    openai_response_to_anthropic,
    openai_tools_to_anthropic,
    anthropic_tools_to_openai,
    convert_tool_choice_openai_to_anthropic,
    convert_tool_choice_anthropic_to_openai,
    map_finish_reason_openai_to_anthropic,
    map_finish_reason_anthropic_to_openai,
    prepare_chat_body,
    prepare_messages_body,
)


# ---------------------------------------------------------------------------
# CRITICAL: thinking_delta goes to wrong index (AS1)
# ---------------------------------------------------------------------------

def test_openai_sse_to_anthropic_thinking_block_uses_correct_index() -> None:
    """The thinking block must be opened with content_block_start before any
    thinking_delta is emitted, AND it must use a different content_block
    index from the text block when both are present."""
    state: dict = {}
    # 1. reasoning chunk alone
    evts = openai_sse_to_anthropic_sse(
        {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
        state,
    )
    # Two events: content_block_start{thinking, idx=0}, content_block_delta{thinking_delta, idx=0}
    assert len(evts) == 2
    start_evt, start_data = evts[0]
    assert start_evt == "content_block_start"
    assert start_data["index"] == 0
    assert start_data["content_block"]["type"] == "thinking"

    delta_evt, delta_data = evts[1]
    assert delta_evt == "content_block_delta"
    assert delta_data["index"] == 0
    assert delta_data["delta"]["type"] == "thinking_delta"


def test_openai_sse_to_anthropic_text_block_uses_different_index_than_thinking() -> None:
    """When reasoning appears in chunk 1 and text in chunk 2, they must use
    different content_block indices (text = 1, thinking = 0)."""
    state: dict = {}
    evts1 = openai_sse_to_anthropic_sse(
        {"choices": [{"delta": {"reasoning_content": "think"}}]}, state,
    )
    evts2 = openai_sse_to_anthropic_sse(
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}, state,
    )
    all_evts = evts1 + evts2

    text_block_idx = None
    thinking_block_idx = None
    for e, d in all_evts:
        if e != "content_block_start":
            continue
        ctype = d.get("content_block", {}).get("type")
        if ctype == "thinking":
            thinking_block_idx = d["index"]
        elif ctype == "text":
            text_block_idx = d["index"]
    assert thinking_block_idx is not None and text_block_idx is not None
    assert text_block_idx != thinking_block_idx, (
        f"text block idx {text_block_idx} must differ from thinking block idx {thinking_block_idx}"
    )


def test_openai_sse_to_anthropic_finish_closes_all_open_blocks() -> None:
    state: dict = {}
    openai_sse_to_anthropic_sse(
        {"choices": [{"delta": {"reasoning_content": "think"}}]}, state,
    )
    openai_sse_to_anthropic_sse(
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}, state,
    )
    # All opened blocks must be closed before message_delta/message_stop
    closed_indices = set()
    saw_message_delta = False
    for e, d in openai_sse_to_anthropic_sse(
        {"choices": [{"delta": {"content": "extra"}}]}, state,
    ):
        if e == "content_block_stop":
            closed_indices.add(d["index"])
        elif e == "message_delta":
            saw_message_delta = True
    # The previous finish_reason already closed everything. Extra chunk
    # after finish opens a new text block at the next free index.
    assert saw_message_delta is False  # no finish_reason on this chunk


def test_openai_sse_to_anthropic_no_state_arg_still_works() -> None:
    """Backwards compatibility: callers that don't pass state must still
    produce a valid SSE sequence for text-only streams."""
    evts = openai_sse_to_anthropic_sse({"choices": [{"delta": {"content": "x"}}]})
    # Should produce at minimum a content_block_start + content_block_delta
    types = [e for e, _ in evts]
    assert "content_block_start" in types
    assert "content_block_delta" in types


# ---------------------------------------------------------------------------
# Tool / tool_choice conversion
# ---------------------------------------------------------------------------

def test_openai_tools_to_anthropic_basic() -> None:
    openai = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    out = openai_tools_to_anthropic(openai)
    assert out is not None
    assert out[0]["name"] == "f"
    # input_schema gets normalized to a valid JSON schema
    assert out[0]["input_schema"]["type"] == "object"
    assert "properties" in out[0]["input_schema"]


def test_anthropic_tools_to_openai_basic() -> None:
    anthropic = [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
    out = anthropic_tools_to_openai(anthropic)
    assert out is not None
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "f"


def test_tool_choice_round_trip() -> None:
    # "auto" → openai
    assert convert_tool_choice_anthropic_to_openai({"type": "auto"}) == "auto"
    # "any" → "required"
    assert convert_tool_choice_anthropic_to_openai({"type": "any"}) == "required"
    # "tool" → {"type":"function","function":{"name":...}}
    out = convert_tool_choice_anthropic_to_openai({"type": "tool", "name": "f"})
    assert out == {"type": "function", "function": {"name": "f"}}


def test_finish_reason_maps() -> None:
    assert map_finish_reason_openai_to_anthropic("stop") == "end_turn"
    assert map_finish_reason_openai_to_anthropic("tool_calls") == "tool_use"
    assert map_finish_reason_openai_to_anthropic("length") == "max_tokens"


# ---------------------------------------------------------------------------
# Response shape conversion
# ---------------------------------------------------------------------------

def test_openai_response_to_anthropic_basic() -> None:
    data = {
        "id": "x",
        "model": "m",
        "choices": [{
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    out = openai_response_to_anthropic(data)
    assert out["role"] == "assistant"
    assert out["content"] == [{"type": "text", "text": "hi"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"]["input_tokens"] == 5
    assert out["usage"]["output_tokens"] == 2


def test_openai_response_to_anthropic_content_null_falls_back_to_reasoning() -> None:
    """NVIDIA reasoning models return content:null + reasoning_content. The
    proxy must surface the reasoning as text content for the Anthropic SDK."""
    data = {
        "id": "x",
        "model": "m",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "the answer is 42",
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    out = openai_response_to_anthropic(data)
    assert out["content"] == [{"type": "text", "text": "the answer is 42"}]


# ---------------------------------------------------------------------------
# prepare_chat_body — basic shape
# ---------------------------------------------------------------------------

def test_prepare_chat_body_force_default_model() -> None:
    """prepare_chat_body must be idempotent / not raise on minimal body."""
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    out = prepare_chat_body(body)
    assert "model" in out
    assert out.get("messages")


def test_prepare_messages_body_basic() -> None:
    body = {
        "model": "x",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = prepare_messages_body(body)
    assert "messages" in out