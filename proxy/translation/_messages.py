"""Bidirectional conversion of messages arrays between Anthropic and OpenAI.

These are the largest pure functions in the translation package. They
handle system-prompt extraction, content-block flattening, tool_use /
tool_result round-tripping, image_url <-> image-block conversion, and
preservation of Anthropic-private keys (cache_control, thinking blocks)
that need to round-trip back to Anthropic.

The two functions are colocated because they share the same private-key
vocabulary and the same edge-case table.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..utils import loads
from ._helpers import stringify_args

# Re-bind for legacy import path (this module used to be inlined in
# proxy/translation.py and callers referenced the underscore names).
_stringify_args = stringify_args


def anthropic_messages_to_openai(
    messages: List[Dict],
    system: Any = None,
) -> List[Dict]:
    """Best-effort Anthropic messages (+ system) -> OpenAI chat messages."""
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
                                "arguments": stringify_args(block.get("input", {})),
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
                                "arguments": stringify_args(block.get("input", {})),
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
                        "content": stringify_args(c)
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
                                {"type": "text", "text": stringify_args(ob)}
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
            content = "\n".join(texts) if texts else stringify_args(content)
        out.append({"role": role, "content": content})
    return out


def openai_messages_to_anthropic(
    messages: List[Dict],
) -> Tuple[List[Dict], Optional[str]]:
    """OpenAI chat messages -> Anthropic messages + optional system string."""
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
            # Also pick up cache_control stashed from a prior Anthropic->OpenAI round-trip
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
                    c = stringify_args(c)
                tr: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or "",
                    "content": c,
                }
                if isinstance(c, str) and c.startswith("[error]"):
                    tr["is_error"] = True
                    tr["content"] = c[len("[error]"):].strip()
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
