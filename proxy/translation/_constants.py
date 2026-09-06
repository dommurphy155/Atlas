"""Module-level constants used across the translation sub-modules.

Centralised here so the request/response/sse converters can all share the
same allow-lists, strip-lists, and effort/thinking vocabularies without
each importing from the others (avoids circular imports).
"""
from __future__ import annotations

from typing import Dict, FrozenSet

# OpenAI / OpenRouter reasoning_effort values (plus common aliases).
_EFFORT_VALUES: FrozenSet[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)

# Map common non-standard effort strings -> canonical.
_EFFORT_ALIASES: Dict[str, str] = {
    "auto": "medium",
    "default": "medium",
    "normal": "medium",
    "full": "high",
    "maximum": "max",
    "ultra": "high",
    "min": "minimal",
    "disabled": "none",
    "off": "none",
    "0": "none",
    "1": "low",
    "2": "medium",
    "3": "high",
}

# Anthropic thinking.type values we understand.
_THINKING_TYPES: FrozenSet[str] = frozenset({"enabled", "adaptive", "disabled"})

# Content-block types that indicate an Anthropic-shaped message list.
_ANTHROPIC_BLOCK_TYPES: FrozenSet[str] = frozenset(
    {
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
        "image",
        "document",
        "search_result",
        "server_tool_use",
        "web_search_tool_result",
        "code_execution_tool_result",
        "mcp_tool_use",
        "mcp_tool_result",
        "container_upload",
        "text",  # alone is ambiguous; used with others
    }
)

# Top-level keys that are known to be invalid / dangerous for OpenRouter
# chat/completions or messages and should be stripped after normalization.
# (Responses-API-only fields, harness-private keys, deprecated aliases that
# we have already rewritten, etc.)
_STRIP_AFTER_NORMALIZE: FrozenSet[str] = frozenset(
    {
        # OpenAI Responses API shapes (not supported on chat/messages path)
        "input",
        "instructions",
        "previous_response_id",
        "response_id",
        "conversation",
        "conversation_id",
        "parent_response_id",
        "store",
        "truncation",
        "include",
        "text",  # Responses structured-output container
        # Already consumed / rewritten
        "thinking",
        "reasoning_budget",
        "thinking_budget",
        "budget_tokens",
        "include_reasoning",
        "reasoning_tokens",
        "reasoning_mode",
        "reasoning_details",  # request-side only; response path is separate
        # Common harness private / experimental keys
        "betas",
        "anthropic_beta",
        "anthropic_version",
        "x-api-key",
        "api_key",
        "extra_headers",
        "extra_body",
        "client",
        "timeout",
        "http_client",
        "base_url",
        "default_headers",
        "_debug_render_only",
        "debug_render_only",
    }
)

# Keys that are safe to forward unchanged (OpenRouter accepts them or
# silently ignores them without 400).
_SAFE_FORWARD: FrozenSet[str] = frozenset(
    {
        "model",
        "models",
        "messages",
        "system",
        "prompt",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stop",
        "stop_sequences",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "top_a",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "seed",
        "n",
        "stream",
        "stream_options",
        "response_format",
        "structured_outputs",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "user",
        "metadata",
        "provider",
        "plugins",
        "transforms",
        "route",
        "reasoning",
        "reasoning_effort",
        "verbosity",
        "output_config",
        "modalities",
        "prediction",
        "service_tier",
        "web_search_options",
        "cache_control",
        "max_tool_calls",
        "stop_server_tools_when",
    }
)
