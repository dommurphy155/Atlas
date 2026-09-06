"""Request-body preparation: prepare_chat_body and prepare_messages_body.

These are the two public entry points that every incoming /v1/chat/completions
or /v1/messages request flows through before being sent upstream. Both
share a common pipeline:

  1. model resolution (force-default or pass-through)
  2. tools / tool_choice conversion
  3. messages array conversion (if mixed shape detected)
  4. stop / max_tokens alias unification
  5. reasoning / thinking normalization
  6. strip keys that the upstream provider rejects
  7. inject system prompt override
  8. enforce input token limit (strip thinking, trim oldest)
  9. strip internal private keys before upstream

Most of the helpers in this module are private (underscore prefix) and
shared with the truncation helpers in the lower half of the file.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import get_logger
from ..system_prompt import _inject_system_override_anthropic, _inject_system_override_openai
from ._constants import _STRIP_AFTER_NORMALIZE
from ._helpers import messages_look_anthropic, messages_look_openai_tools
from ._reasoning import (
    collect_reasoning_hints,
    emit_reasoning_for_openai_chat,
    emit_thinking_for_anthropic,
)
from ._tools import (
    anthropic_tools_to_openai,
    convert_tool_choice_anthropic_to_openai,
    convert_tool_choice_openai_to_anthropic,
    openai_tools_to_anthropic,
)

# Re-bind private names so the rest of the file can use the old local
# import path (this module was inlined in proxy/translation.py before).
_collect_reasoning_hints = collect_reasoning_hints
_emit_reasoning_for_openai_chat = emit_reasoning_for_openai_chat
_emit_thinking_for_anthropic = emit_thinking_for_anthropic
_messages_look_anthropic = messages_look_anthropic
_messages_look_openai_tools = messages_look_openai_tools

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Token estimation + trimming
# ---------------------------------------------------------------------------

# Rough char->token ratio for heuristic estimation. Good enough for trimming
# decisions; we don't need precision, just a consistent ordering.
_CHAR_PER_TOKEN = 3.6


def _estimate_tokens(text: Optional[str]) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / _CHAR_PER_TOKEN))


def _message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate token count for a single message, including nested content."""
    total = 0
    content = msg.get("content")
    role = msg.get("role", "")
    total += _estimate_tokens(role)  # role token overhead

    if isinstance(content, str):
        total += _estimate_tokens(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" or "text" in block:
                total += _estimate_tokens(block.get("text", ""))
            if "input" in block and isinstance(block["input"], str):
                total += _estimate_tokens(block["input"])
            if "source" in block and isinstance(block["source"], dict):
                src = block["source"]
                if src.get("type") == "base64" and src.get("data"):
                    # Base64 data — rough estimate (data is ~3/4 of original bytes)
                    total += len(src["data"]) * 3 // 4 // int(_CHAR_PER_TOKEN)
            # Strip thinking from token count if present (internal monologue)
            if btype == "thinking" and block.get("thinking"):
                total += _estimate_tokens(block.get("thinking", ""))
    # Tool calls
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            total += _estimate_tokens(fn.get("name", ""))
            total += _estimate_tokens(fn.get("arguments", ""))
    return total


def _trim_message_content(msg: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    """Truncate a single message's content to fit within max_tokens.

    For string content, truncates the text. For list content (Anthropic blocks),
    truncates the last text block. Returns a new dict.
    """
    msg = dict(msg)
    content = msg.get("content")
    if isinstance(content, str):
        # Reserve ~10% for role/other overhead
        allowed_chars = int(max_tokens * _CHAR_PER_TOKEN * 0.9)
        if len(content) > allowed_chars:
            msg["content"] = content[:allowed_chars] + "...[truncated]"
    elif isinstance(content, list):
        # Truncate text blocks from the end until under budget
        remaining = max_tokens
        new_blocks = []
        for block in reversed(content):
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue
            bt = block.get("type")
            if bt == "text":
                ttokens = _estimate_tokens(block.get("text", ""))
                if ttokens > remaining:
                    allowed_chars = int(remaining * _CHAR_PER_TOKEN * 0.9)
                    block = {"type": "text", "text": block.get("text", "")[:allowed_chars] + "...[truncated]"}
                    new_blocks.append(block)
                    remaining = 0
                else:
                    new_blocks.append(block)
                    remaining -= ttokens
            else:
                # Keep non-text blocks (images, tool_use) — expensive but usually few
                ttokens = _message_tokens({"content": [block]})
                remaining -= ttokens
                new_blocks.append(block)
        msg["content"] = list(reversed(new_blocks))
        if remaining < 0:
            # Still over — we did our best with text truncation
            pass
    return msg


def _trim_messages(messages: List[Dict], max_tokens: int) -> List[Dict]:
    """Trim oldest user/assistant/tool messages when total tokens exceed max.

    Keeps system/developer messages intact. Truncates individual message
    content as a last resort rather than dropping messages entirely.
    """
    if not messages or max_tokens <= 0:
        return messages

    total = sum(_message_tokens(m) for m in messages)
    if total <= max_tokens:
        return messages

    # Split into protected (system/developer) and trimmable (user/assistant/tool)
    protected = []
    trimmable = []
    for m in messages:
        role = m.get("role", "") if isinstance(m, dict) else "user"
        if role in ("system", "developer"):
            protected.append(m)
        else:
            trimmable.append(m)

    # Trim from the front of trimmable list (oldest first)
    while len(trimmable) > 1 and total > max_tokens:
        removed = trimmable.pop(0)
        total -= _message_tokens(removed)

    # If a single trimmable message is still over budget, truncate its content
    if total > max_tokens and trimmable:
        remaining_budget = max_tokens - sum(_message_tokens(m) for m in protected)
        if remaining_budget > 0:
            # Truncate each remaining message proportionally
            trimmable = [
                _trim_message_content(m, remaining_budget // len(trimmable))
                for m in trimmable
            ]

    # Rebuild: protected messages stay in their original relative positions
    # but trimmable ones that remain are appended after. This is a best-effort
    # reordering — Anthropic/OpenRouter accept system at any position.
    result = []
    remaining = list(trimmable)
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        if role in ("system", "developer"):
            result.append(m)
            if m in remaining:
                remaining.remove(m)
    result.extend(remaining)
    return result


def _strip_internal_from_messages(messages: List[Dict]) -> List[Dict]:
    """Remove thinking/reasoning blocks from assistant messages before trimming.

    These are internal monologue tokens that consume budget without adding
    value for the upstream model's context. Claude Code includes them in
    the conversation history, but they're not useful for the model to re-read
    — they're output, not input.
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                b for b in content
                if isinstance(b, dict) and b.get("type") != "thinking" and b.get("type") != "redacted_thinking"
            ]
    return messages


def _enforce_input_limit(messages: List[Dict]) -> List[Dict]:
    """Strip internal monologue and trim to MAX_INPUT_TOKENS."""
    from ..config import MAX_INPUT_TOKENS
    messages = _strip_internal_from_messages(messages)
    return _trim_messages(messages, MAX_INPUT_TOKENS)


_PRIVATE_KEYS = frozenset(
    {"_anthropic_thinking_blocks", "_anthropic_cache_control"}
)


def _strip_private_keys(messages: List[Dict]) -> List[Dict]:
    """Remove internal private keys from messages before sending to upstream.

    These keys (e.g. _anthropic_cache_control, _anthropic_thinking_blocks) are
    used to carry Anthropic-specific metadata through OpenAI-format conversion
    for round-trip preservation, but must not reach the upstream provider.
    """
    for msg in messages:
        if isinstance(msg, dict):
            for k in _PRIVATE_KEYS:
                msg.pop(k, None)
    return messages


def _sanitize_body(body: Dict[str, Any], *, target: str) -> Dict[str, Any]:
    """Drop keys that are known to cause provider 400s after we have
    normalized the ones we understand.  Keep everything in _SAFE_FORWARD
    and any other key that looks like a simple scalar/plugin option.
    """
    # Always remove the aliases we have already consumed
    for k in list(body.keys()):
        if k in _STRIP_AFTER_NORMALIZE:
            body.pop(k, None)
    return body


def _apply_max_tokens_alias(body: Dict[str, Any], *, prefer_completion: bool = False) -> None:
    """Unify max_tokens / max_completion_tokens."""
    mt = body.get("max_tokens")
    mct = body.get("max_completion_tokens")
    if prefer_completion:
        if mct is None and mt is not None:
            body["max_completion_tokens"] = mt
        # leave both; OpenRouter accepts either
    else:
        if mt is None and mct is not None:
            body["max_tokens"] = mct


# ---------------------------------------------------------------------------
# Public prepare functions
# ---------------------------------------------------------------------------

def prepare_chat_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an incoming chat/completions body for OpenRouter.

    Accepts pure OpenAI, pure Anthropic, or mixed harness payloads.
    Emits only OpenRouter-safe chat/completions fields.
    """
    from ..config import get_force_default_model, get_default_model

    # Work on a shallow copy so we never mutate the caller's dict
    body = dict(body)

    # ----- Codex / empty request handling -----
    if not body.get("messages") and not body.get("prompt"):
        session_id = body.get("client_metadata", {}).get("session_id", "unknown")
        body["messages"] = [{"role": "user", "content": f"[Codex session {session_id}]"}]

    # Model resolution: per-provider force flag (default True) overrides every
    # request to the provider's default model. When False, the client's model
    # passes through if present (for multi-model setups).
    if get_force_default_model() or not body.get("model"):
        body["model"] = get_default_model()

    # ----- tools -----
    if "tools" in body:
        body["tools"] = anthropic_tools_to_openai(body["tools"])

    # ----- tool_choice -----
    if "tool_choice" in body:
        tc = body["tool_choice"]
        # Fold disable_parallel_tool_use (Anthropic) into parallel_tool_calls
        # (OpenAI). Without this, an Anthropic client that explicitly requests
        # "no parallel tool calls" silently gets the OpenAI default of
        # parallel_tool_calls=True.
        if isinstance(tc, dict) and tc.get("disable_parallel_tool_use"):
            body["parallel_tool_calls"] = False
        body["tool_choice"] = convert_tool_choice_anthropic_to_openai(tc)

    # ----- messages (heuristic conversion) -----
    msgs = body.get("messages")
    if msgs and isinstance(msgs, list) and msgs:
        # Convert when any Anthropic-style block is present (not only first msg)
        if messages_look_anthropic(msgs):
            system = body.pop("system", None)
            body["messages"] = anthropic_messages_to_openai(msgs, system)
        # else: leave pure OpenAI messages alone

    # ----- stop -----
    if "stop_sequences" in body and "stop" not in body:
        body["stop"] = body.pop("stop_sequences")
    elif "stop_sequences" in body and "stop" in body:
        # Prefer the OpenAI key; drop the Anthropic alias
        body.pop("stop_sequences", None)

    # ----- max_tokens alias -----
    _apply_max_tokens_alias(body, prefer_completion=False)

    # ----- reasoning / thinking normalization -----
    hints = collect_reasoning_hints(body)
    if hints:
        reasoning_obj, effort_shorthand = emit_reasoning_for_openai_chat(hints)
        # Remove all source aliases first
        for k in (
            "thinking",
            "reasoning",
            "reasoning_effort",
            "reasoning_budget",
            "thinking_budget",
            "budget_tokens",
            "include_reasoning",
            "verbosity",
        ):
            body.pop(k, None)
        if reasoning_obj is not None:
            body["reasoning"] = reasoning_obj
        if effort_shorthand is not None and "reasoning" not in body:
            body["reasoning_effort"] = effort_shorthand
        # Keep verbosity only if it was not consumed as effort
        # (already popped above)

    # ----- parallel_tool_calls stays as-is (OpenRouter understands it) -----

    # ----- strip dangerous / Responses-only keys -----
    body = _sanitize_body(body, target="openai")

    # ----- system prompt override (additive) -----
    if "messages" in body and isinstance(body["messages"], list):
        body["messages"] = _inject_system_override_openai(body["messages"])

    # ----- enforce input token limit (strip thinking, trim oldest) -----
    if "messages" in body and isinstance(body["messages"], list):
        body["messages"] = _enforce_input_limit(body["messages"])

    # ----- strip internal private keys from messages before upstream -----
    if "messages" in body and isinstance(body["messages"], list):
        body["messages"] = _strip_private_keys(body["messages"])

    return body


def prepare_messages_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an incoming Anthropic /messages body for OpenRouter.

    Accepts pure Anthropic, pure OpenAI, or mixed harness payloads.
    Emits only OpenRouter-safe Anthropic-messages fields.
    """
    from ..config import get_force_default_model, get_default_model

    body = dict(body)

    # Same model resolution policy as prepare_chat_body: force to default
    # when per-provider flag is set or model is absent; otherwise pass through.
    if get_force_default_model() or not body.get("model"):
        body["model"] = get_default_model()

    # Capture parallel_tool_calls before we may strip it
    parallel = body.get("parallel_tool_calls")

    # ----- tools -----
    if "tools" in body:
        body["tools"] = openai_tools_to_anthropic(body["tools"])

    # ----- tool_choice (also folds parallel_tool_calls) -----
    if "tool_choice" in body or parallel is not None:
        tc = body.get("tool_choice")
        body["tool_choice"] = convert_tool_choice_openai_to_anthropic(
            tc, parallel_tool_calls=parallel
        )

    # ----- messages -----
    msgs = body.get("messages")
    if msgs and isinstance(msgs, list):
        if messages_look_openai_tools(msgs) or any(
            isinstance(m, dict) and m.get("role") in ("system", "developer", "tool")
            for m in msgs
        ):
            # Also convert when system/developer roles are present so they
            # become the top-level system field.
            converted, system = openai_messages_to_anthropic(msgs)
            body["messages"] = converted
            if system and "system" not in body:
                body["system"] = system
        # else: pure Anthropic messages left alone

    # ----- stop -----
    if "stop" in body and "stop_sequences" not in body:
        stop = body.pop("stop")
        if isinstance(stop, str):
            body["stop_sequences"] = [stop]
        elif isinstance(stop, list):
            body["stop_sequences"] = stop
    elif "stop" in body and "stop_sequences" in body:
        body.pop("stop", None)

    # ----- max_tokens (Anthropic requires it; alias from max_completion_tokens) -----
    _apply_max_tokens_alias(body, prefer_completion=False)
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, (int, float)):
        max_tokens = int(max_tokens)
    else:
        max_tokens = None

    # ----- reasoning / thinking normalization -----
    hints = collect_reasoning_hints(body)
    if hints:
        thinking_obj, output_config = emit_thinking_for_anthropic(
            hints, max_tokens=max_tokens
        )
        for k in (
            "thinking",
            "reasoning",
            "reasoning_effort",
            "reasoning_budget",
            "thinking_budget",
            "budget_tokens",
            "include_reasoning",
            "verbosity",
        ):
            body.pop(k, None)
        if thinking_obj is not None:
            body["thinking"] = thinking_obj
        if output_config is not None:
            existing_oc = body.get("output_config")
            if isinstance(existing_oc, dict):
                existing_oc = dict(existing_oc)
                existing_oc.update(output_config)
                body["output_config"] = existing_oc
            else:
                body["output_config"] = output_config

        # Anthropic constraint: thinking + forced tool_choice is illegal
        if thinking_obj and thinking_obj.get("type") in ("enabled", "adaptive"):
            tc = body.get("tool_choice")
            if isinstance(tc, dict) and tc.get("type") in ("any", "tool"):
                log.warning(
                    "Dropping forced tool_choice %s because thinking is enabled "
                    "(Anthropic rejects this combination)",
                    tc,
                )
                body["tool_choice"] = {"type": "auto"}
            elif tc == "required":
                body["tool_choice"] = {"type": "auto"}

    # ----- strip parallel_tool_calls (folded into tool_choice) -----
    body.pop("parallel_tool_calls", None)

    # ----- strip dangerous keys -----
    body = _sanitize_body(body, target="anthropic")

    # ----- system prompt override -----
    if "messages" in body and isinstance(body["messages"], list):
        body["messages"], body["system"] = _inject_system_override_anthropic(
            body["messages"], body.get("system")
        )

    # ----- enforce input token limit (strip thinking, trim oldest) -----
    if "messages" in body and isinstance(body["messages"], list):
        body["messages"] = _enforce_input_limit(body["messages"])

    # ----- strip internal private keys from messages before upstream -----
    if "messages" in body and isinstance(body["messages"], list):
        body["messages"] = _strip_private_keys(body["messages"])

    return body


# Also re-export the message converters under their original import names so
# `from ._requests import anthropic_messages_to_openai` works for any caller
# that imported them from proxy.translation.
from ._messages import anthropic_messages_to_openai, openai_messages_to_anthropic  # noqa: E402
