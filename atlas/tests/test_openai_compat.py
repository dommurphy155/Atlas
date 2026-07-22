"""openai_compat — payload sanitization + OpenAI↔Anthropic shape translation.

These are the highest-regression-risk pure functions: a bad clamp lets NVIDIA
400 the request, a bad message translation drops tool calls and breaks Claude
Code, a bad SSE translation corrupts the stream. All deterministic, no network.
"""
from __future__ import annotations

import json

import pytest

from proxy.openai_compat import (
    _clamp,
    anthropic_messages_to_openai,
    anthropic_openai_payload,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    normalize_messages,
    openai_response_to_anthropic,
    sanitize_openai_payload,
)


# ── sanitize_openai_payload ──────────────────────────────────────────────────

class TestSanitize:
    def test_clamps_temperature_into_range(self):
        out = sanitize_openai_payload({"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 2.5})
        assert out["temperature"] == 2.0

    def test_clamps_temperature_floor(self):
        out = sanitize_openai_payload({"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": -1})
        assert out["temperature"] == 0.0

    def test_clamps_top_p(self):
        out = sanitize_openai_payload({"model": "m", "messages": [{"role": "user", "content": "hi"}], "top_p": 2})
        assert out["top_p"] == 1.0

    def test_clamps_penalties(self):
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 9, "presence_penalty": -9,
        })
        assert out["frequency_penalty"] == 2.0
        assert out["presence_penalty"] == -2.0

    def test_drops_unsupported_fields(self):
        """Allowlist must drop OpenAI-only fields GLM rejects (top_k, seed, n,
        logprobs, reasoning_effort, ...)."""
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "top_k": 40, "seed": 7, "n": 3, "logprobs": True,
            "reasoning_effort": "high", "user": "u", "logit_bias": {},
        })
        for dropped in ("top_k", "seed", "n", "logprobs", "reasoning_effort", "user", "logit_bias"):
            assert dropped not in out, f"{dropped} should be dropped"

    def test_max_tokens_default_when_missing(self):
        out = sanitize_openai_payload({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert out["max_tokens"] == 1024

    def test_max_completion_tokens_honored(self):
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 500,
        })
        assert out["max_tokens"] == 500

    def test_stream_options_added_only_when_streaming(self):
        out = sanitize_openai_payload({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True})
        assert out["stream"] is True
        assert out["stream_options"] == {"include_usage": True}

    def test_no_stream_options_when_not_streaming(self):
        out = sanitize_openai_payload({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert "stream_options" not in out

    def test_tools_passthrough_with_auto_choice(self):
        out = sanitize_openai_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        })
        assert out["tools"][0]["function"]["name"] == "f"
        assert out["tool_choice"] == "auto"

    def test_stop_string_normalized_to_list(self):
        out = sanitize_openai_payload({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stop": "END"})
        assert out["stop"] == ["END"]


def test_clamp_nan_and_inf_fall_back():
    assert _clamp(float("nan"), 0.0, 1.0, 0.5) == 0.5
    assert _clamp(float("inf"), 0.0, 1.0, 0.5) == 0.5
    assert _clamp("not-a-number", 0.0, 1.0, 0.5) == 0.5


# ── normalize_messages ───────────────────────────────────────────────────────

class TestNormalize:
    def test_flattens_text_parts(self):
        out = normalize_messages([
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        ])
        assert out[0]["content"] == "a\nb"

    def test_preserves_assistant_tool_calls(self):
        tc = [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}]
        out = normalize_messages([{"role": "assistant", "content": "ok", "tool_calls": tc}])
        assert out[0]["tool_calls"] == tc

    def test_preserves_tool_role_call_id(self):
        out = normalize_messages([{"role": "tool", "content": "result", "tool_call_id": "call_1", "name": "f"}])
        assert out[0]["tool_call_id"] == "call_1"
        assert out[0]["name"] == "f"

    def test_empty_messages_rejected(self):
        with pytest.raises(ValueError):
            normalize_messages([])

    def test_non_object_message_rejected(self):
        with pytest.raises(ValueError):
            normalize_messages(["not a dict"])

    def test_none_content_becomes_empty_string(self):
        out = normalize_messages([{"role": "user", "content": None}])
        assert out[0]["content"] == ""


# ── anthropic → openai conversion ────────────────────────────────────────────

class TestAnthropicToOpenAI:
    def test_system_field_becomes_system_message(self):
        out = anthropic_messages_to_openai({"system": "you are X", "messages": [{"role": "user", "content": "hi"}]})
        assert out[0] == {"role": "system", "content": "you are X"}
        assert out[1]["role"] == "user"

    def test_system_list_of_blocks(self):
        out = anthropic_messages_to_openai({
            "system": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert out[0]["content"] == "part1\npart2"

    def test_assistant_tool_use_becomes_tool_calls(self):
        out = anthropic_messages_to_openai({"messages": [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking..."},
                {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "x"}},
            ],
        }]})
        asst = out[0]
        assert asst["role"] == "assistant"
        assert asst["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}

    def test_tool_result_becomes_tool_role(self):
        out = anthropic_messages_to_openai({"messages": [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "42"}]},
        ]})
        tool_msg = [m for m in out if m["role"] == "tool"][0]
        assert tool_msg["tool_call_id"] == "tu1"
        assert tool_msg["content"] == "42"

    def test_thinking_blocks_dropped(self):
        out = anthropic_messages_to_openai({"messages": [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "text": "internal"},
                {"type": "text", "text": "visible"},
            ],
        }]})
        assert out[0]["content"] == "visible"

    def test_tools_translated(self):
        tools = [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
        out = anthropic_tools_to_openai(tools)
        assert out[0]["function"]["name"] == "f"
        assert out[0]["function"]["parameters"] == {"type": "object"}

    def test_tool_choice_translation(self):
        assert anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"
        assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"
        assert anthropic_tool_choice_to_openai({"type": "tool", "name": "f"}) == {"type": "function", "function": {"name": "f"}}
        assert anthropic_tool_choice_to_openai("weird") is None

    def test_payload_sanitizes_temperature(self):
        """anthropic_openai_payload must clamp an out-of-range Anthropic temp
        before it hits NVIDIA (the bug the sanitizer exists to prevent)."""
        body = {"messages": [{"role": "user", "content": "hi"}], "temperature": 5.0}
        out = anthropic_openai_payload(body, "z-ai/glm-5.2")
        assert out["temperature"] == 2.0


# ── openai → anthropic response ──────────────────────────────────────────────

class TestOpenAIToAnthropic:
    def test_text_response(self):
        openai = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
        out = openai_response_to_anthropic("claude", openai)
        assert out["content"][0] == {"type": "text", "text": "hello"}
        assert out["stop_reason"] == "end_turn"
        assert out["usage"] == {"input_tokens": 5, "output_tokens": 3}

    def test_tool_calls_response(self):
        openai = {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}},
        ]}, "finish_reason": "tool_calls"}]}
        out = openai_response_to_anthropic("claude", openai)
        assert out["stop_reason"] == "tool_use"
        assert out["content"][0]["type"] == "tool_use"
        assert out["content"][0]["input"] == {"a": 1}

    def test_empty_response_falls_back_to_router_content(self):
        openai = {"choices": [{"message": {}}]}
        out = openai_response_to_anthropic("claude", openai)
        # Falls back to extract_router_content which json-dumps the payload.
        assert out["content"][0]["type"] == "text"
