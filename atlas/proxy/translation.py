"""Bidirectional OpenAI ↔ Anthropic protocol translation."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from .utils import loads
from .system_prompt import _inject_system_override_openai, _inject_system_override_anthropic
from .config import SYSTEM_PROMPT_OVERRIDE, get_logger

log = get_logger(__name__)


def _stringify_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        import orjson

        return orjson.dumps(args).decode()
    except Exception:
        return str(args)


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
        first = msgs[0] if isinstance(msgs[0], dict) else {}
        first_content = first.get("content")
        if isinstance(first_content, list) and any(
            isinstance(b, dict)
            and b.get("type")
            in (
                "tool_use",
                "tool_result",
                "thinking",
                "redacted_thinking",
                "image",
            )
            for b in first_content
        ):
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
