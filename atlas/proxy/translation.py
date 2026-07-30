"""Bidirectional OpenAI ↔ Anthropic protocol translation + streaming."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, TypedDict, Literal

import httpx
import orjson

from .utils import loads
from .system_prompt import _inject_system_override_openai, _inject_system_override_anthropic
from .config import SYSTEM_PROMPT_OVERRIDE, get_logger

log = get_logger(__name__)


# =============================================================================
# TypedDict definitions for type safety and exhaustive matching
# =============================================================================

class OpenAIChatMessage(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | List[Any] | None
    tool_calls: List[Dict]
    tool_call_id: str
    name: str


class AnthropicMessage(TypedDict, total=False):
    role: Literal["user", "assistant"]
    content: str | List[Dict]


class OpenAITool(TypedDict):
    type: Literal["function"]
    function: Dict[str, Any]


class AnthropicTool(TypedDict, total=False):
    name: str
    input_schema: Dict[str, Any]
    description: str


class OpenAIToolChoice(TypedDict, total=False):
    type: Literal["auto", "required", "none", "function"]
    function: Dict[str, str]


class AnthropicToolChoice(TypedDict, total=False):
    type: Literal["auto", "any", "none", "tool"]
    name: str


class OpenAIReasoning(TypedDict, total=False):
    effort: Literal["low", "medium", "high", "none"]
    max_tokens: int


class AnthropicThinking(TypedDict, total=False):
    type: Literal["enabled", "adaptive", "disabled"]
    budget_tokens: int


# Streaming event types
class OpenAIStreamDelta(TypedDict, total=False):
    role: str
    content: str
    tool_calls: List[Dict]


class OpenAIStreamChoice(TypedDict, total=False):
    index: int
    delta: OpenAIStreamDelta
    finish_reason: Optional[str]


class OpenAIStreamChunk(TypedDict, total=False):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIStreamChoice]


class AnthropicStreamEvent(TypedDict, total=False):
    type: Literal["message_start", "message_delta", "message_stop", "content_block_start", "content_block_delta", "content_block_stop"]
    message: Dict
    delta: Dict
    index: int
    content_block: Dict


def _stringify_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        import orjson
        return orjson.dumps(args).decode()
    except Exception:
        return str(args)


def _has_anthropic_format(messages: List[Dict]) -> bool:
    """Check if ANY message has Anthropic-style content blocks (tool_use, tool_result, thinking, etc.)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "tool_use", "tool_result", "thinking", "redacted_thinking", "image"
                ):
                    return True
    return False


def _has_tool_role(messages: List[Dict]) -> bool:
    """Check if any message has role='tool' (OpenAI tool result format)."""
    return any(msg.get("role") == "tool" for msg in messages)


def anthropic_tools_to_openai(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """Anthropic {name, description, input_schema} → OpenAI function tools."""
    if not tools:
        return tools
    out: List[Dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and "function" in t:
            out.append(t)
            continue
        name = t.get("name") or ""
        desc = t.get("description")
        schema = t.get("input_schema") or t.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        fn: Dict[str, Any] = {"name": name, "parameters": schema}
        if desc is not None:
            fn["description"] = desc
        if "strict" in t:
            fn["strict"] = t["strict"]
        out.append({"type": "function", "function": fn})
    return out


def openai_tools_to_anthropic(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """OpenAI function tools → Anthropic {name, description, input_schema}."""
    if not tools:
        return tools
    out: List[Dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if "input_schema" in t and "name" in t and "function" not in t:
            out.append(t)
            continue
        fn = t.get("function") or t
        name = fn.get("name") or t.get("name") or ""
        desc = fn.get("description")
        params = fn.get("parameters") or fn.get("input_schema") or {
            "type": "object",
            "properties": {},
        }
        tool: Dict[str, Any] = {"name": name, "input_schema": params}
        if desc is not None:
            tool["description"] = desc
        if "strict" in fn:
            tool["strict"] = fn["strict"]
        out.append(tool)
    return out


def convert_tool_choice_anthropic_to_openai(tc: Any) -> Any:
    if tc is None:
        return None
    if isinstance(tc, str):
        return {"auto": "auto", "any": "required", "none": "none"}.get(tc, tc)
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "auto":
            return "auto"
        if t == "any":
            return "required"
        if t == "none":
            return "none"
        if t == "tool" and "name" in tc:
            return {"type": "function", "function": {"name": tc["name"]}}
        if t == "function":
            return tc
    return tc


def convert_tool_choice_openai_to_anthropic(tc: Any) -> Any:
    if tc is None:
        return None
    if tc == "auto":
        return {"type": "auto"}
    if tc == "required":
        return {"type": "any"}
    if tc == "none":
        return {"type": "none"}
    if isinstance(tc, dict):
        if tc.get("type") == "function":
            name = (tc.get("function") or {}).get("name")
            if name:
                return {"type": "tool", "name": name}
        if "type" in tc:
            return tc
    return tc


def anthropic_messages_to_openai(
    messages: List[Dict],
    system: Any = None,
) -> List[Dict]:
    """Best-effort Anthropic messages (+ system) → OpenAI chat messages."""
    out: List[Dict] = []
    if system:
        if isinstance(system, list):
            text = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in system
            )
        else:
            text = str(system)
        if text.strip():
            out.append({"role": "system", "content": text})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "assistant" and isinstance(content, list):
            text_parts: List[str] = []
            tool_calls: List[Dict] = []
            thinking_parts: List[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": _stringify_args(block.get("input", {})),
                            },
                        }
                    )
                elif btype == "thinking":
                    t = block.get("thinking") or block.get("text") or ""
                    if t:
                        thinking_parts.append(t)
                elif btype == "redacted_thinking":
                    thinking_parts.append(block.get("data") or "[redacted_thinking]")
            oai: Dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) or None,
            }
            if tool_calls:
                oai["tool_calls"] = tool_calls
            if thinking_parts:
                oai["reasoning_content"] = "\n".join(thinking_parts)
            out.append(oai)
            continue

        if role == "user" and isinstance(content, list):
            text_parts = []
            tool_results: List[Dict] = []
            image_parts: List[Dict] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    c = block.get("content", "")
                    if isinstance(c, list):
                        c = "\n".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in c
                        )
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id")
                            or block.get("id")
                            or "",
                            "content": _stringify_args(c)
                            if not isinstance(c, str)
                            else c,
                        }
                    )
                    if block.get("is_error"):
                        tool_results[-1]["content"] = (
                            f"[error] {tool_results[-1]['content']}"
                        )
                elif btype == "image":
                    src = block.get("source") or {}
                    if src.get("type") == "base64":
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{src.get('media_type', 'image/png')}"
                                        f";base64,{src.get('data', '')}"
                                    ),
                                },
                            }
                        )
                    elif src.get("type") == "url":
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": src.get("url", "")},
                            }
                        )
            for tr in tool_results:
                out.append(tr)
            if text_parts or image_parts:
                if image_parts:
                    content_list: List[Any] = [
                        {"type": "text", "text": t} for t in text_parts
                    ] + image_parts
                    out.append({"role": "user", "content": content_list})
                else:
                    out.append({"role": "user", "content": "\n".join(text_parts)})
            continue

        if isinstance(content, list):
            texts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "\n".join(texts) if texts else _stringify_args(content)
        out.append({"role": role, "content": content})
    return out


def openai_messages_to_anthropic(
    messages: List[Dict],
) -> Tuple[List[Dict], Optional[str]]:
    """OpenAI chat messages → Anthropic messages + optional system string."""
    system_parts: List[str] = []
    out: List[Dict] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        role = msg.get("role")
        content = msg.get("content")

        if role == "system" or role == "developer":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        system_parts.append(b.get("text", ""))
            i += 1
            continue

        if role == "assistant":
            blocks: List[Dict] = []
            rc = msg.get("reasoning_content") or msg.get("reasoning")
            if rc:
                blocks.append({"type": "thinking", "thinking": str(rc)})
            for rd in msg.get("reasoning_details") or []:
                if isinstance(rd, dict) and rd.get("text"):
                    blocks.append({"type": "thinking", "thinking": rd["text"]})
                elif isinstance(rd, str):
                    blocks.append({"type": "thinking", "thinking": rd})

            if content:
                if isinstance(content, str):
                    blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            blocks.append(
                                {"type": "text", "text": b.get("text", "")}
                            )
                        elif isinstance(b, dict):
                            blocks.append(b)
                        else:
                            blocks.append({"type": "text", "text": str(b)})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = loads(args)
                    except Exception:
                        args = {"raw": args}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )
            out.append({"role": "assistant", "content": blocks or (content or "")})
            i += 1
            continue

        if role == "tool":
            blocks = []
            while i < n and messages[i].get("role") == "tool":
                m = messages[i]
                c = m.get("content")
                if not isinstance(c, (str, list)):
                    c = _stringify_args(c)
                tr: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or "",
                    "content": c,
                }
                if isinstance(c, str) and c.startswith("[error]"):
                    tr["is_error"] = True
                    tr["content"] = c[len("[error]") :].strip()
                blocks.append(tr)
                i += 1
            out.append({"role": "user", "content": blocks})
            continue

        if isinstance(content, list):
            anthro_blocks: List[Dict] = []
            for part in content:
                if not isinstance(part, dict):
                    anthro_blocks.append({"type": "text", "text": str(part)})
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    anthro_blocks.append(
                        {"type": "text", "text": part.get("text", "")}
                    )
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        try:
                            header, b64 = url.split(",", 1)
                            media = header.split(";")[0].split(":")[1]
                            anthro_blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media,
                                        "data": b64,
                                    },
                                }
                            )
                        except Exception:
                            anthro_blocks.append({"type": "text", "text": url})
                    else:
                        anthro_blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": url},
                            }
                        )
                else:
                    anthro_blocks.append(part)
            out.append({"role": role or "user", "content": anthro_blocks})
        else:
            out.append({"role": role or "user", "content": content or ""})
        i += 1

    system = "\n".join(system_parts) if system_parts else None
    return out, system


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


def _has_anthropic_format(messages: list) -> bool:
    """Check if any message has Anthropic-style content blocks (tool_use, tool_result, thinking, image)."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "tool_use", "tool_result", "thinking", "redacted_thinking", "image"
                ):
                    return True
    return False


def _has_tool_role(messages: list) -> bool:
    """Check if any message has role='tool' (OpenAI tool result format)."""
    return any(isinstance(m, dict) and m.get("role") == "tool" for m in messages)


def prepare_chat_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an incoming chat/completions body for OpenRouter."""
    from .config import FORCE_DEFAULT_MODEL, OPENROUTER_MODEL

    if FORCE_DEFAULT_MODEL or not body.get("model"):
        body["model"] = OPENROUTER_MODEL

    if "tools" in body:
        body["tools"] = anthropic_tools_to_openai(body["tools"])
    if "tool_choice" in body:
        body["tool_choice"] = convert_tool_choice_anthropic_to_openai(
            body["tool_choice"]
        )

    msgs = body.get("messages")
    if msgs and isinstance(msgs, list) and msgs:
        # M5 FIX: Scan ALL messages for Anthropic format, not just first
        if _has_anthropic_format(msgs) or _has_tool_role(msgs):
            system = body.pop("system", None)
            body["messages"] = anthropic_messages_to_openai(msgs, system)

    if "thinking" in body and "reasoning" not in body:
        thinking = body.pop("thinking")
        if isinstance(thinking, dict):
            ttype = thinking.get("type")
            if ttype in ("enabled", "adaptive"):
                budget = thinking.get("budget_tokens")
                if budget:
                    body["reasoning"] = {"max_tokens": budget}
                else:
                    body["reasoning"] = {"effort": "high"}
            elif ttype == "disabled":
                body["reasoning"] = {"effort": "none"}

    if "stop_sequences" in body and "stop" not in body:
        body["stop"] = body.pop("stop_sequences")

    # Inject system prompt override (additive, non-destructive)
    if "messages" in body and isinstance(body["messages"], list):
        body["messages"] = _inject_system_override_openai(body["messages"])

    return body


# =============================================================================
# OpenAI Responses API ↔ Anthropic Messages translation
# =============================================================================

def openai_responses_to_anthropic(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI Responses API format to Anthropic Messages format."""
    messages = []

    # Handle 'input' field (string, list of strings, or list of dicts)
    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content") or item.get("text", "")
                messages.append({"role": role, "content": content})

    # Handle 'instructions' → system prompt
    system = body.get("instructions", "")

    # Tools
    tools = body.get("tools")
    if tools:
        tools = openai_tools_to_anthropic(tools)

    # Tool choice
    tool_choice = body.get("tool_choice")
    if tool_choice:
        tool_choice = convert_tool_choice_openai_to_anthropic(tool_choice)

    # Reasoning → thinking
    thinking = None
    if "reasoning" in body:
        reasoning = body["reasoning"]
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            max_tok = reasoning.get("max_tokens")
            if effort in ("none", None) and not max_tok:
                thinking = {"type": "disabled"}
            elif max_tok:
                thinking = {"type": "enabled", "budget_tokens": max_tok}
            else:
                thinking = {"type": "adaptive"}
                if effort and effort not in ("none",):
                    thinking["effort"] = effort

    # Stop sequences
    stop_sequences = None
    if "stop" in body:
        stop = body["stop"]
        if isinstance(stop, str):
            stop_sequences = [stop]
        elif isinstance(stop, list):
            stop_sequences = stop

    return {
        "model": body.get("model", "nemotron-3-ultra-550b-a55b:free"),
        "messages": messages,
        "system": system if system else None,
        "tools": tools,
        "tool_choice": tool_choice,
        "thinking": thinking,
        "stop_sequences": stop_sequences,
        "stream": body.get("stream", False),
        "max_tokens": body.get("max_output_tokens", 4096),
        "temperature": body.get("temperature", 1.0),
        "top_p": body.get("top_p", 1.0),
    }


def anthropic_to_openai_responses(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic Messages format to OpenAI Responses API format."""
    input_items = []

    # Convert messages to input format
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            input_items.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Handle Anthropic content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        input_items.append({
                            "type": "function_call",
                            "call_id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        })
                    elif btype == "tool_result":
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": block.get("tool_use_id", ""),
                            "output": block.get("content", ""),
                        })
            if text_parts:
                input_items.append({"role": role, "content": "\n".join(text_parts)})

    # System → instructions
    instructions = None
    system = body.get("system")
    if isinstance(system, list):
        instructions = "\n".join(b.get("text", "") for b in system if b.get("type") == "text")
    elif isinstance(system, str):
        instructions = system

    # Tools
    tools = body.get("tools")
    if tools:
        tools = anthropic_tools_to_openai(tools)

    # Tool choice
    tool_choice = body.get("tool_choice")
    if tool_choice:
        tool_choice = convert_tool_choice_anthropic_to_openai(tool_choice)

    # Thinking → reasoning
    reasoning = None
    thinking = body.get("thinking")
    if thinking and isinstance(thinking, dict):
        ttype = thinking.get("type")
        if ttype == "enabled":
            reasoning = {"max_tokens": thinking.get("budget_tokens", 4096)}
        elif ttype == "adaptive":
            reasoning = {"effort": "high"}
        elif ttype == "disabled":
            reasoning = {"effort": "none"}

    return {
        "model": body.get("model", "nemotron-3-ultra-550b-a55b:free"),
        "input": input_items,
        "instructions": instructions,
        "tools": tools,
        "tool_choice": tool_choice,
        "reasoning": reasoning,
        "stream": body.get("stream", False),
        "max_output_tokens": body.get("max_tokens", 4096),
        "temperature": body.get("temperature", 1.0),
        "top_p": body.get("top_p", 1.0),
    }


def prepare_messages_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an incoming Anthropic /messages body for OpenRouter."""
    from .config import FORCE_DEFAULT_MODEL, OPENROUTER_MODEL

    if FORCE_DEFAULT_MODEL or not body.get("model"):
        body["model"] = OPENROUTER_MODEL

    if "tools" in body:
        body["tools"] = openai_tools_to_anthropic(body["tools"])
    if "tool_choice" in body:
        body["tool_choice"] = convert_tool_choice_openai_to_anthropic(
            body["tool_choice"]
        )

    msgs = body.get("messages")
    if msgs and isinstance(msgs, list):
        has_tool_role = any(
            isinstance(m, dict) and m.get("role") == "tool" for m in msgs
        )
        has_tool_calls = any(
            isinstance(m, dict) and "tool_calls" in m for m in msgs
        )
        if has_tool_role or has_tool_calls:
            converted, system = openai_messages_to_anthropic(msgs)
            body["messages"] = converted
            if system and "system" not in body:
                body["system"] = system

    if "reasoning" in body and "thinking" not in body:
        reasoning = body.pop("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            max_tok = reasoning.get("max_tokens")
            if effort in ("none", None) and not max_tok:
                body["thinking"] = {"type": "disabled"}
            elif max_tok:
                body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": max_tok,
                }
            else:
                body["thinking"] = {"type": "adaptive"}
                if effort and effort not in ("none",):
                    body.setdefault("output_config", {})["effort"] = effort

    if "stop" in body and "stop_sequences" not in body:
        stop = body.pop("stop")
        if isinstance(stop, str):
            body["stop_sequences"] = [stop]
        elif isinstance(stop, list):
            body["stop_sequences"] = stop

    # Inject system prompt override for Anthropic format (additive, non-destructive)
    if "messages" in body and isinstance(body["messages"], list):
        body["messages"], body["system"] = _inject_system_override_anthropic(
            body["messages"], body.get("system")
        )

    return body


# =============================================================================
# Streaming Translation (SSE frames both ways)
# =============================================================================

async def translate_stream_openai_to_anthropic(
    upstream_response: httpx.Response,
) -> AsyncIterator[bytes]:
    """
    Convert OpenAI SSE stream → Anthropic SSE stream.

    OpenAI: data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}
    Anthropic: event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}
    """
    import httpx
    import orjson

    content_block_index = 0
    message_started = False

    try:
        async for raw in upstream_response.aiter_raw():
            if not raw:
                continue

            for frame in raw.split(b"\n\n"):
                frame = frame.strip()
                if not frame or frame == b"data: [DONE]":
                    continue

                if frame.startswith(b"data: "):
                    try:
                        data = orjson.loads(frame[6:])
                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        finish_reason = choices[0].get("finish_reason")

                        if not message_started:
                            # Send message_start
                            yield b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stream\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"stream\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n"
                            message_started = True

                            # Send content_block_start
                            yield b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"

                        # Handle content delta
                        if "content" in delta and delta["content"]:
                            yield orjson.dumps({
                                "type": "content_block_delta",
                                "index": content_block_index,
                                "delta": {"type": "text_delta", "text": delta["content"]}
                            }).decode().encode() + b"\n\n"

                        # Handle tool calls
                        if "tool_calls" in delta and delta["tool_calls"]:
                            for tc in delta["tool_calls"]:
                                fn = tc.get("function", {})
                                yield orjson.dumps({
                                    "type": "content_block_start",
                                    "index": content_block_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                                        "name": fn.get("name", ""),
                                        "input": fn.get("arguments", "{}")
                                    }
                                }).decode().encode() + b"\n\n"
                                content_block_index += 1

                        # Handle finish
                        if finish_reason:
                            stop_reason = "end_turn"
                            if finish_reason == "tool_calls":
                                stop_reason = "tool_use"
                            elif finish_reason == "length":
                                stop_reason = "max_tokens"

                            yield b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
                            yield orjson.dumps({
                                "type": "message_delta",
                                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                "usage": {"output_tokens": 0}
                            }).decode().encode() + b"\n\n"
                            yield b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                            return

                    except orjson.JSONDecodeError:
                        continue

    except (httpx.ReadError, httpx.StreamError, asyncio.CancelledError):
        pass
    except Exception as e:
        log.warning(f"Stream translation error: {e}")


async def translate_stream_anthropic_to_openai(
    upstream_response: httpx.Response,
) -> AsyncIterator[bytes]:
    """
    Convert Anthropic SSE stream → OpenAI SSE stream.

    Anthropic: event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}
    OpenAI: data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}
    """
    import httpx
    import orjson

    try:
        async for raw in upstream_response.aiter_raw():
            if not raw:
                continue

            for frame in raw.split(b"\n\n"):
                frame = frame.strip()
                if not frame:
                    continue

                # Parse SSE frame
                event_type = None
                data = None

                for line in frame.split(b"\n"):
                    line = line.strip()
                    if line.startswith(b"event: "):
                        event_type = line[7:].decode()
                    elif line.startswith(b"data: "):
                        try:
                            data = orjson.loads(line[6:])
                        except orjson.JSONDecodeError:
                            continue

                if not data:
                    continue

                # Translate Anthropic events to OpenAI format
                if event_type == "message_start":
                    # Could send role delta here
                    yield b"data: " + orjson.dumps({
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "stream",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                    }) + b"\n\n"

                elif event_type == "content_block_start":
                    block = data.get("content_block", {})
                    if block.get("type") == "tool_use":
                        yield b"data: " + orjson.dumps({
                            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "stream",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "id": block.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name", ""),
                                            "arguments": ""
                                        }
                                    }]
                                },
                                "finish_reason": None
                            }]
                        }) + b"\n\n"

                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield b"data: " + orjson.dumps({
                                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "stream",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None
                                }]
                            }) + b"\n\n"
                    elif delta.get("type") == "input_json_delta":
                        # Tool call arguments streaming
                        partial = delta.get("partial_json", "")
                        if partial:
                            yield b"data: " + orjson.dumps({
                                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "stream",
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{
                                            "index": 0,
                                            "function": {"arguments": partial}
                                        }]
                                    },
                                    "finish_reason": None
                                }]
                            }) + b"\n\n"

                elif event_type == "message_delta":
                    delta = data.get("delta", {})
                    if delta.get("stop_reason"):
                        stop_reason = delta["stop_reason"]
                        finish_reason = "stop"
                        if stop_reason == "tool_use":
                            finish_reason = "tool_calls"
                        elif stop_reason == "max_tokens":
                            finish_reason = "length"
                        yield b"data: " + orjson.dumps({
                            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "stream",
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason
                            }]
                        }) + b"\n\n"

                elif event_type == "message_stop":
                    yield b"data: [DONE]\n\n"
                    return

    except (httpx.ReadError, httpx.StreamError, asyncio.CancelledError):
        pass
    except Exception as e:
        log.warning(f"Stream translation error: {e}")
