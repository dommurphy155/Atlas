"""Bidirectional OpenAI ↔ Anthropic protocol translation.

Goal: accept every reasonable harness / SDK payload shape and emit only
fields that OpenRouter (and the underlying providers it routes to) will
accept. Unknown or provider-rejected keys are normalized or dropped so
they never produce upstream HTTP 400s.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from .utils import loads
from .system_prompt import _inject_system_override_openai, _inject_system_override_anthropic
from .config import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OpenAI / OpenRouter reasoning_effort values (plus common aliases).
_EFFORT_VALUES = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)

# Map common non-standard effort strings → canonical.
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
_THINKING_TYPES = frozenset({"enabled", "adaptive", "disabled"})

# Content-block types that indicate an Anthropic-shaped message list.
_ANTHROPIC_BLOCK_TYPES = frozenset(
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
_STRIP_AFTER_NORMALIZE = frozenset(
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
_SAFE_FORWARD = frozenset(
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stringify_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        import orjson

        return orjson.dumps(args).decode()
    except Exception:
        return str(args)


def _normalize_effort(value: Any) -> Optional[str]:
    """Map any effort-like value to a canonical OpenRouter effort string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Numeric budgets sometimes arrive as effort; treat large numbers as high.
        if value <= 0:
            return "none"
        if value < 1024:
            return "low"
        if value < 8192:
            return "medium"
        if value < 32768:
            return "high"
        return "xhigh"
    s = str(value).strip().lower()
    if not s:
        return None
    if s in _EFFORT_VALUES:
        return s
    if s in _EFFORT_ALIASES:
        return _EFFORT_ALIASES[s]
    # Unknown effort string — log a warning so the silent fallback to "medium"
    # is visible, then clamp to a safe default rather than passing through
    # something OpenRouter may reject.
    log.warning("Unknown reasoning_effort value %r — defaulting to 'medium'", s)
    return "medium"


def _extract_budget(obj: Any) -> Optional[int]:
    """Pull a positive integer token budget from assorted shapes."""
    if obj is None:
        return None
    if isinstance(obj, (int, float)) and obj > 0:
        return int(obj)
    if isinstance(obj, dict):
        for key in (
            "budget_tokens",
            "max_tokens",
            "thinking_budget",
            "reasoning_budget",
            "tokens",
            "budget",
        ):
            v = obj.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
    return None


def _is_anthropic_content_list(content: Any) -> bool:
    if not isinstance(content, list) or not content:
        return False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in _ANTHROPIC_BLOCK_TYPES and btype != "text":
            return True
        # Anthropic image / tool_result without explicit type still detectable
        if "tool_use_id" in block or "input_schema" in block:
            return True
        if isinstance(block.get("source"), dict) and block["source"].get("type") in (
            "base64",
            "url",
        ):
            return True
    return False


def _messages_look_anthropic(messages: List[Any]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if _is_anthropic_content_list(content):
            return True
        # Anthropic never uses role "tool"; presence of tool_use blocks already caught.
    return False


def _messages_look_openai_tools(messages: List[Any]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            return True
        if "tool_calls" in msg:
            return True
    return False


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def anthropic_tools_to_openai(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """Anthropic {name, description, input_schema} → OpenAI function tools."""
    if not tools:
        return tools
    out: List[Dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        # Already OpenAI chat-shaped: {"type": "function", "function": {...}}
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            out.append(t)
            continue
        # OpenRouter server tools passthrough (never have a "function" key)
        if t.get("type") in ("openrouter:web_search", "openrouter:datetime",
                             "openrouter:image_generation", "openrouter:web_fetch",
                             "openrouter:apply_patch", "openrouter:shell", "openrouter:fusion"):
            out.append(t)
            continue
        # Codex custom/freeform tools are not OpenAI function tools.
        # OpenRouter needs a JSON-callable representation, so adapt every
        # custom/freeform tool generically. The original tool type is preserved
        # request-locally by routes.py and restored on the Responses API side.
        #
        # A freeform/custom grammar has no JSON parameter schema, therefore the
        # adapter exposes one string field, `input`, containing the raw tool
        # input. This is an internal transport representation only.
        if t.get("type") in ("freeform", "custom"):
            name = t.get("name")
            if not name:
                continue

            original_desc = t.get("description") or ""
            desc = (
                original_desc
                + ("\n\n" if original_desc else "")
                + "This is a custom/freeform tool. "
                  "Arguments MUST be a JSON object with a single string field "
                  "`input` containing the complete raw tool input."
            )

            out.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Complete raw input for the custom/freeform tool.",
                            },
                        },
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                },
            })
            continue

        # Hosted custom/freeform tool forms may arrive as a bare tool type with
        # no `name`. Preserve the known hosted form as a generic custom tool.
        if t.get("type") == "apply_patch" and "name" not in t:
            out.append({
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": (
                        "Custom/freeform tool. Arguments MUST be a JSON object "
                        "with a single string field `input` containing the complete "
                        "raw tool input."
                    ),
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"},
                        },
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                },
            })
            continue
        # Anthropic custom / server tool passthrough when it already has type
        if "type" in t and t["type"] not in (None, "function") and "name" not in t:
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
        # Preserve cache_control for round-trip back to Anthropic format.
        # OpenAI chat/completions doesn't support it natively; we stash it
        # on the function dict under a private key.
        if "cache_control" in t:
            fn["_anthropic_cache_control"] = t["cache_control"]
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
        # Already Anthropic-shaped
        if "input_schema" in t and "name" in t and "function" not in t:
            out.append(t)
            continue
        # OpenRouter server tools – leave as-is; Anthropic path may not understand them
        ttype = t.get("type")
        if isinstance(ttype, str) and ttype.startswith("openrouter:"):
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
        # Re-emit cache_control preserved from anthropic_tools_to_openai
        if "_anthropic_cache_control" in fn:
            tool["cache_control"] = fn["_anthropic_cache_control"]
        out.append(tool)
    return out


# ---------------------------------------------------------------------------
# tool_choice
# ---------------------------------------------------------------------------

def convert_tool_choice_anthropic_to_openai(tc: Any) -> Any:
    if tc is None:
        return None
    if isinstance(tc, str):
        return {"auto": "auto", "any": "required", "none": "none", "required": "required"}.get(
            tc, tc
        )
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
        # Preserve disable_parallel_tool_use → parallel_tool_calls is handled elsewhere
    return tc


def convert_tool_choice_openai_to_anthropic(
    tc: Any, *, parallel_tool_calls: Optional[bool] = None
) -> Any:
    if tc is None:
        return None
    disable_parallel = parallel_tool_calls is False

    if tc == "auto":
        out: Dict[str, Any] = {"type": "auto"}
        if disable_parallel:
            out["disable_parallel_tool_use"] = True
        return out
    if tc == "required":
        out = {"type": "any"}
        if disable_parallel:
            out["disable_parallel_tool_use"] = True
        return out
    if tc == "none":
        return {"type": "none"}
    if isinstance(tc, dict):
        if tc.get("type") == "function":
            name = (tc.get("function") or {}).get("name")
            if name:
                out = {"type": "tool", "name": name}
                if disable_parallel:
                    out["disable_parallel_tool_use"] = True
                return out
        if tc.get("type") in ("auto", "any", "none", "tool"):
            out = dict(tc)
            if disable_parallel and "disable_parallel_tool_use" not in out:
                out["disable_parallel_tool_use"] = True
            return out
        if "type" in tc:
            return tc
    return tc


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

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
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "assistant" and isinstance(content, list):
            text_parts: List[str] = []
            tool_calls: List[Dict] = []
            thinking_parts: List[str] = []
            # Preserve cache_control on text blocks for round-trip back to
            # Anthropic format. OpenAI chat/completions doesn't support it
            # natively; we stash it on a private key.
            cache_controls: List[Dict[str, Any]] = []
            # Preserve signature-bearing thinking blocks for later round-trip
            # by stuffing them into a private key that most OpenAI clients ignore.
            raw_thinking_blocks: List[Dict] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                    cc = block.get("cache_control")
                    if cc:
                        cache_controls.append(cc)
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
                    if "cache_control" in block:
                        cache_controls.append(block["cache_control"])
                elif btype == "thinking":
                    t = block.get("thinking") or block.get("text") or ""
                    if t:
                        thinking_parts.append(t)
                    raw_thinking_blocks.append(block)
                elif btype == "redacted_thinking":
                    thinking_parts.append(block.get("data") or "[redacted_thinking]")
                    raw_thinking_blocks.append(block)
                elif btype in ("server_tool_use", "mcp_tool_use"):
                    # Best-effort: treat as ordinary tool call
                    tool_calls.append(
                        {
                            "id": block.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": block.get("name", btype),
                                "arguments": _stringify_args(block.get("input", {})),
                            },
                        }
                    )
                    if "cache_control" in block:
                        cache_controls.append(block["cache_control"])
            oai: Dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) or None,
            }
            if tool_calls:
                oai["tool_calls"] = tool_calls
            if thinking_parts:
                oai["reasoning_content"] = "\n".join(thinking_parts)
            if raw_thinking_blocks:
                # Non-standard but harmless; allows later Anthropic round-trip
                oai["_anthropic_thinking_blocks"] = raw_thinking_blocks
            if cache_controls:
                oai["_anthropic_cache_control"] = cache_controls
            out.append(oai)
            continue

        if role == "user" and isinstance(content, list):
            text_parts = []
            tool_results: List[Dict] = []
            image_parts: List[Dict] = []
            other_parts: List[Any] = []
            user_cache_controls: List[Dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    if block is not None:
                        text_parts.append(str(block))
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                    cc = block.get("cache_control")
                    if cc:
                        user_cache_controls.append(cc)
                elif btype == "tool_result":
                    c = block.get("content", "")
                    if isinstance(c, list):
                        c = "\n".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in c
                        )
                    tr = {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id")
                        or block.get("id")
                        or "",
                        "content": _stringify_args(c)
                        if not isinstance(c, str)
                        else c,
                    }
                    if block.get("is_error"):
                        tr["content"] = f"[error] {tr['content']}"
                    if "cache_control" in block:
                        tr["_anthropic_cache_control"] = block["cache_control"]
                    tool_results.append(tr)
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
                else:
                    # Preserve unknown blocks as text fallback
                    other_parts.append(block)
            for tr in tool_results:
                out.append(tr)
            if text_parts or image_parts or other_parts:
                if image_parts or other_parts:
                    content_list: List[Any] = [
                        {"type": "text", "text": t} for t in text_parts
                    ] + image_parts
                    # Attach unknown blocks as text so they are not silently lost
                    for ob in other_parts:
                        if isinstance(ob, dict) and ob.get("type") == "text":
                            content_list.append(ob)
                        else:
                            content_list.append(
                                {"type": "text", "text": _stringify_args(ob)}
                            )
                    user_msg: Dict[str, Any] = {"role": "user", "content": content_list}
                    if user_cache_controls:
                        user_msg["_anthropic_cache_control"] = user_cache_controls
                    out.append(user_msg)
                else:
                    user_msg2: Dict[str, Any] = {"role": "user", "content": "\n".join(text_parts)}
                    if user_cache_controls:
                        user_msg2["_anthropic_cache_control"] = user_cache_controls
                    out.append(user_msg2)
            continue

        # Fallback: flatten list content to string
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
    system_cache_controls: List[Dict[str, Any]] = []
    out: List[Dict] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if not isinstance(msg, dict):
            i += 1
            continue
        role = msg.get("role")
        content = msg.get("content")

        if role == "system" or role == "developer":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        system_parts.append(b.get("text", ""))
                        if "cache_control" in b:
                            system_cache_controls.append(b["cache_control"])
                    elif isinstance(b, str):
                        system_parts.append(b)
            # Also pick up cache_control stashed from a prior Anthropic→OpenAI round-trip
            cc = msg.get("_anthropic_cache_control")
            if isinstance(cc, list):
                system_cache_controls.extend(cc)
            i += 1
            continue

        if role == "assistant":
            blocks: List[Dict] = []
            # Prefer preserved Anthropic thinking blocks when present
            preserved = msg.get("_anthropic_thinking_blocks")
            if isinstance(preserved, list) and preserved:
                for pb in preserved:
                    if isinstance(pb, dict):
                        blocks.append(pb)
            else:
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
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = loads(args)
                    except Exception:
                        args = {"raw": args}
                tblock: Dict[str, Any] = {
                    "type": "tool_use",
                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": fn.get("name", ""),
                    "input": args if isinstance(args, dict) else {"raw": args},
                }
                if "_anthropic_cache_control" in fn:
                    tblock["cache_control"] = fn["_anthropic_cache_control"]
                blocks.append(tblock)
            # Re-emit cache_control on text blocks from round-trip private key
            cc_list = msg.get("_anthropic_cache_control")
            if isinstance(cc_list, list) and cc_list:
                cc_idx = 0
                for blk in blocks:
                    if isinstance(blk, dict) and blk.get("type") == "text" and cc_idx < len(cc_list):
                        blk["cache_control"] = cc_list[cc_idx]
                        cc_idx += 1
            out.append({"role": "assistant", "content": blocks or (content or "")})
            i += 1
            continue

        if role == "tool":
            blocks = []
            while i < n and isinstance(messages[i], dict) and messages[i].get("role") == "tool":
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
                cc = m.get("_anthropic_cache_control")
                if cc is not None:
                    # Handle both dict and list storage formats
                    tr["cache_control"] = cc[0] if isinstance(cc, list) and cc else cc
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
            # Re-emit cache_control from round-trip private key
            cc_list = msg.get("_anthropic_cache_control")
            if isinstance(cc_list, list) and cc_list:
                cc_idx = 0
                for blk in anthro_blocks:
                    if isinstance(blk, dict) and blk.get("type") == "text" and cc_idx < len(cc_list):
                        blk["cache_control"] = cc_list[cc_idx]
                        cc_idx += 1
            out.append({"role": role or "user", "content": anthro_blocks})
        else:
            user_out: Dict[str, Any] = {"role": role or "user", "content": content or ""}
            cc_list = msg.get("_anthropic_cache_control")
            if isinstance(cc_list, list) and cc_list and content:
                # Wrap string content in a block list to carry cache_control
                user_out["content"] = [
                    {"type": "text", "text": content, "cache_control": cc_list[0]}
                ]
            out.append(user_out)
        i += 1

    if system_parts and system_cache_controls:
        # Return as block list to preserve cache_control on system blocks
        system: Any = [
            {"type": "text", "text": system_parts[0], "cache_control": cc}
            if idx < len(system_cache_controls)
            else {"type": "text", "text": part}
            for idx, part in enumerate(system_parts)
        ]
    elif system_parts:
        system = "\n".join(system_parts)
    else:
        system = None
    return out, system


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
# Reasoning / thinking normalization (core of the 10/10 fix)
# ---------------------------------------------------------------------------

def _collect_reasoning_hints(body: Dict[str, Any]) -> Dict[str, Any]:
    """Gather every known reasoning/thinking alias into a single internal form.

    Returns a dict that may contain:
      mode: "none" | "effort" | "budget" | "adaptive"
      effort: str
      budget_tokens: int
      exclude: bool
      display: str (Anthropic)
    """
    hints: Dict[str, Any] = {}

    # 1. Explicit reasoning object (OpenRouter / OpenAI style)
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        if "effort" in reasoning:
            e = _normalize_effort(reasoning["effort"])
            if e:
                hints["effort"] = e
                hints["mode"] = "effort" if e != "none" else "none"
        if "max_tokens" in reasoning:
            b = _extract_budget(reasoning)
            if b:
                hints["budget_tokens"] = b
                hints.setdefault("mode", "budget")
        if reasoning.get("exclude") is True or reasoning.get("exclude") == "true":
            hints["exclude"] = True
        if reasoning.get("enabled") is False:
            hints["mode"] = "none"
        elif reasoning.get("enabled") is True and "mode" not in hints:
            hints["mode"] = "adaptive"

    # 2. Top-level reasoning_effort (OpenAI / OpenRouter shorthand)
    if "reasoning_effort" in body:
        e = _normalize_effort(body["reasoning_effort"])
        if e:
            hints["effort"] = e
            hints["mode"] = "effort" if e != "none" else "none"

    # 3. Anthropic thinking object
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        ttype = thinking.get("type")
        if ttype == "disabled":
            hints["mode"] = "none"
        elif ttype == "enabled":
            b = _extract_budget(thinking)
            if b:
                hints["budget_tokens"] = max(b, 1024)  # Anthropic minimum
                hints["mode"] = "budget"
            else:
                hints["mode"] = "adaptive"
        elif ttype == "adaptive":
            hints["mode"] = "adaptive"
        if "display" in thinking:
            hints["display"] = thinking["display"]
    elif isinstance(thinking, bool):
        hints["mode"] = "adaptive" if thinking else "none"
    elif isinstance(thinking, str):
        t = thinking.lower()
        if t in ("enabled", "on", "true"):
            hints["mode"] = "adaptive"
        elif t in ("disabled", "off", "false", "none"):
            hints["mode"] = "none"

    # 4. Top-level budget aliases
    for key in ("budget_tokens", "thinking_budget", "reasoning_budget"):
        if key in body:
            b = _extract_budget(body[key])
            if b:
                hints["budget_tokens"] = max(b, 1024)
                hints.setdefault("mode", "budget")

    # 5. include_reasoning (deprecated OpenRouter alias → exclude=False)
    if body.get("include_reasoning") is True:
        hints["exclude"] = False
    elif body.get("include_reasoning") is False:
        hints["exclude"] = True

    # 6. verbosity → effort mapping for Anthropic (OpenRouter documents this)
    verbosity = body.get("verbosity")
    if verbosity and "effort" not in hints:
        e = _normalize_effort(verbosity)
        if e and e != "none":
            hints["effort"] = e
            hints.setdefault("mode", "effort")

    # 7. output_config.effort (already Anthropic-shaped)
    oc = body.get("output_config")
    if isinstance(oc, dict) and "effort" in oc and "effort" not in hints:
        e = _normalize_effort(oc["effort"])
        if e:
            hints["effort"] = e
            hints.setdefault("mode", "effort")

    return hints


def _emit_reasoning_for_openai_chat(hints: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """Produce (reasoning object, reasoning_effort shorthand) for chat path."""
    if not hints or hints.get("mode") == "none":
        # Explicitly disable when requested
        if hints.get("mode") == "none":
            return {"effort": "none"}, "none"
        return None, None

    reasoning: Dict[str, Any] = {}
    effort = hints.get("effort")
    budget = hints.get("budget_tokens")

    if effort:
        reasoning["effort"] = effort
    elif budget:
        reasoning["max_tokens"] = budget
    else:
        # adaptive / enabled without specifics → high effort
        reasoning["effort"] = "high"

    if hints.get("exclude") is True:
        reasoning["exclude"] = True

    # Prefer the shorthand when only effort is present (cleaner for OpenRouter)
    if list(reasoning.keys()) == ["effort"]:
        return reasoning, reasoning["effort"]
    return reasoning, None


def _emit_thinking_for_anthropic(
    hints: Dict[str, Any], max_tokens: Optional[int] = None
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Produce (thinking object, output_config) for Anthropic messages path."""
    if not hints:
        return None, None

    mode = hints.get("mode")
    if mode == "none":
        return {"type": "disabled"}, None

    thinking: Dict[str, Any] = {}
    output_config: Optional[Dict] = None

    budget = hints.get("budget_tokens")
    effort = hints.get("effort")

    if mode == "budget" and budget:
        # Clamp budget relative to max_tokens when possible
        if max_tokens and budget >= max_tokens:
            budget = max(1024, max_tokens - 1)
        thinking = {"type": "enabled", "budget_tokens": max(budget, 1024)}
    else:
        # Prefer adaptive for modern models; effort goes into output_config
        thinking = {"type": "adaptive"}
        if effort and effort != "none":
            output_config = {"effort": effort}
        elif mode == "adaptive" and not effort:
            pass  # pure adaptive
        else:
            # fallback: if we only had a vague "enabled", use high effort
            output_config = {"effort": effort or "high"}

    if hints.get("display"):
        thinking["display"] = hints["display"]

    return thinking, output_config


# ---------------------------------------------------------------------------
# Context window safety — proactive trimming before upstream 400s
# ---------------------------------------------------------------------------

# Rough char→token ratio for heuristic estimation. Good enough for trimming
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
    from .config import MAX_INPUT_TOKENS
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
    from .config import get_force_default_model, get_default_model

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
        body["tool_choice"] = convert_tool_choice_anthropic_to_openai(
            body["tool_choice"]
        )

    # ----- messages (heuristic conversion) -----
    msgs = body.get("messages")
    if msgs and isinstance(msgs, list) and msgs:
        # Convert when any Anthropic-style block is present (not only first msg)
        if _messages_look_anthropic(msgs):
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
    hints = _collect_reasoning_hints(body)
    if hints:
        reasoning_obj, effort_shorthand = _emit_reasoning_for_openai_chat(hints)
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
    from .config import get_force_default_model, get_default_model

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
        if _messages_look_openai_tools(msgs) or any(
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
    hints = _collect_reasoning_hints(body)
    if hints:
        thinking_obj, output_config = _emit_thinking_for_anthropic(
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


# ---------------------------------------------------------------------------
# OpenAI → Anthropic response translation (for HF provider on /v1/messages)
# ---------------------------------------------------------------------------

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
    import uuid

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
        "model": data.get("model", ""),
        "content": content_blocks + tool_blocks,
        "stop_reason": _map_openai_finish_reason_to_anthropic(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": translate_usage_openai_to_anthropic(data.get("usage")),
    }
    return out


# ---------------------------------------------------------------------------
# OpenAI SSE → Anthropic SSE streaming translation (for HF on /v1/messages)
# ---------------------------------------------------------------------------

def openai_sse_to_anthropic_sse(
    sse_chunk: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Optional[Dict]]]:
    """Convert a single OpenAI SSE payload into Anthropic SSE events.

    ``state`` (optional) tracks which Anthropic content blocks have already been
    started so this stateless-looking function can emit well-formed
    ``content_block_start`` / ``content_block_stop`` pairs for blocks that
    may not have been pre-allocated. Tracks:

    - ``text_open`` (bool)        : whether the text block is currently open
    - ``text_index`` (int)        : Anthropic index for the text block
    - ``thinking_open`` (bool)    : whether the thinking block is currently open
    - ``thinking_index`` (int)    : Anthropic index for the thinking block
    - ``next_index`` (int)        : next free Anthropic block index
    - ``tool_blocks`` (dict idx->started) : per OpenAI tool index, whether the
                                   Anthropic tool_use block was started
    """

    out: List[Tuple[str, Optional[Dict]]] = []

    if state is None:
        state = {}
    state.setdefault("text_open", False)
    state.setdefault("text_index", -1)
    state.setdefault("thinking_open", False)
    state.setdefault("thinking_index", -1)
    state.setdefault("next_index", 0)
    state.setdefault("tool_blocks", {})

    def _ensure_thinking() -> int:
        if not state["thinking_open"]:
            state["thinking_index"] = state["next_index"]
            state["next_index"] += 1
            state["thinking_open"] = True
            out.append(("content_block_start", {
                "type": "content_block_start",
                "index": state["thinking_index"],
                "content_block": {"type": "thinking", "thinking": ""},
            }))
        return state["thinking_index"]

    def _ensure_text() -> int:
        if not state["text_open"]:
            state["text_index"] = state["next_index"]
            state["next_index"] += 1
            state["text_open"] = True
            out.append(("content_block_start", {
                "type": "content_block_start",
                "index": state["text_index"],
                "content_block": {"type": "text", "text": ""},
            }))
        return state["text_index"]

    choices = sse_chunk.get("choices") or []
    if not choices:
        return out

    choice = choices[0]
    delta = choice.get("delta") or {}
    finish_reason = choice.get("finish_reason")

    # --- reasoning / thinking (must precede text per Anthropic spec) ---
    reasoning = delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        idx = _ensure_thinking()
        out.append(("content_block_delta", {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "thinking_delta", "thinking": reasoning},
        }))

    # --- text content ---
    content = delta.get("content")
    if isinstance(content, str) and content:
        idx = _ensure_text()
        out.append(("content_block_delta", {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "text_delta", "text": content},
        }))

    # --- tool calls ---
    tcs = delta.get("tool_calls") or []

    for tc in tcs:
        if not isinstance(tc, dict):
            continue

        idx = tc.get("index", 0)
        fn = tc.get("function") or {}

        if idx not in state["tool_blocks"]:
            anthropic_idx = state["next_index"]
            state["next_index"] += 1
            state["tool_blocks"][idx] = {
                "anthropic_idx": anthropic_idx,
                "name": "",
                "started": False,
            }

        block = state["tool_blocks"][idx]

        if fn.get("name") and not block["started"]:
            block["started"] = True
            block["name"] = fn["name"]
            out.append(("content_block_start", {
                "type": "content_block_start",
                "index": block["anthropic_idx"],
                "content_block": {
                    "type": "tool_use",
                    "id": tc.get("id") or "",
                    "name": fn["name"],
                    "input": {},
                },
            }))

        arguments = fn.get("arguments")
        if arguments:
            if not isinstance(arguments, str):
                import json
                arguments = json.dumps(arguments)

            out.append(("content_block_delta", {
                "type": "content_block_delta",
                "index": block["anthropic_idx"],
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": arguments,
                },
            }))

    # --- finish ---
    if finish_reason:
        # Close any still-open blocks before the message_stop.
        if state["thinking_open"]:
            out.append(("content_block_stop", {
                "type": "content_block_stop",
                "index": state["thinking_index"],
            }))
            state["thinking_open"] = False
        if state["text_open"]:
            out.append(("content_block_stop", {
                "type": "content_block_stop",
                "index": state["text_index"],
            }))
            state["text_open"] = False
        for idx, block in state["tool_blocks"].items():
            if block["started"]:
                out.append(("content_block_stop", {
                    "type": "content_block_stop",
                    "index": block["anthropic_idx"],
                }))
                block["started"] = False

        out.append(("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason":
                    _map_openai_finish_reason_to_anthropic(finish_reason),
            },
        }))

        out.append(("message_stop", {
            "type": "message_stop",
        }))
    return out

