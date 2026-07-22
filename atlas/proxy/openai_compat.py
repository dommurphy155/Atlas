from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any


# These helpers adapt NVIDIA responses into the OpenAI or Anthropic shapes that
# the local proxy exposes to clients.
def completion_id() -> str:
    return f"chatcmpl_{uuid.uuid4().hex}"


def openai_error(message: str, code: str, status: int) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": code,
            "code": status,
        }
    }


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    # Accept standard OpenAI message arrays while preserving tool protocol
    # fields. Flatten text-part content only; dropping tool_calls here breaks
    # Claude Code and other tool-using clients.
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
            content = "\n".join(parts)
        if content is None:
            content = ""
        normalized_message: dict[str, Any] = {"role": role, "content": str(content)}
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            normalized_message["tool_calls"] = message["tool_calls"]
        if role == "tool" and message.get("tool_call_id"):
            normalized_message["tool_call_id"] = str(message["tool_call_id"])
        if message.get("name"):
            normalized_message["name"] = str(message["name"])
        normalized.append(normalized_message)
    return normalized


def non_stream_response(model: str, content: str, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    finish_reason = "tool_calls" if tool_calls else "stop"
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def openai_response_from_router(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    # Preserve provider-native chat-completion responses, especially tool_calls.
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and isinstance(first.get("message"), dict):
            return {
                "id": str(payload.get("id") or completion_id()),
                "object": "chat.completion",
                "created": int(payload.get("created") or time.time()),
                "model": model,
                "choices": choices,
                "usage": payload.get("usage") or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
    return non_stream_response(model, extract_router_content(payload))


def chunk_payload(model: str, content: str) -> dict[str, Any]:
    return {
        "id": completion_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }


async def sse_from_text(model: str, text: str) -> AsyncIterator[bytes]:
    if text:
        yield f"data: {json.dumps(chunk_payload(model, text), separators=(',', ':'))}\n\n".encode()
    yield b"data: [DONE]\n\n"


def extract_router_content(payload: dict[str, Any]) -> str:
    # Router responses can nest the actual text in a few different places, so
    # walk the common shapes before falling back to the raw payload.
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if content is not None:
                    return str(content)
            text = first.get("text")
            if text is not None:
                return str(text)
    generated = payload.get("generated_text")
    if generated is not None:
        return str(generated)
    return json.dumps(payload)


def anthropic_response(model: str, content: str) -> dict[str, Any]:
    return anthropic_response_from_blocks(model, [{"type": "text", "text": content}], "end_turn")


def anthropic_response_from_blocks(
    model: str,
    content_blocks: list[dict[str, Any]],
    stop_reason: str,
) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def anthropic_system_text(system: Any) -> str:
    if isinstance(system, list):
        return "\n".join(
            str(block.get("text") or "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(system or "")


def anthropic_messages_to_openai(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = anthropic_system_text(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty array")

    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role") or "user")
        content = message.get("content", "")

        # Preserve mid-conversation role:system messages (e.g. the end-of-
        # conversation reinforcement injected by replace_system_prompt) as
        # role:system rather than collapsing them to role:user. GLM-5.2 honors
        # a system message anywhere in the thread, and keeping the role lets
        # the reinforcement land as an instruction instead of a fake user turn.
        if role == "system":
            sys_text = content
            if isinstance(sys_text, list):
                sys_text = "\n".join(
                    str(b.get("text") or "") for b in sys_text
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            messages.append({"role": "system", "content": str(sys_text or "")})
            continue

        if not isinstance(content, list):
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": str(content)})
            continue

        blocks = [block for block in content if isinstance(block, dict) and block.get("type") != "thinking"]
        text_parts = [str(block.get("text") or "") for block in blocks if block.get("type") == "text"]
        tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
        tool_results = [block for block in blocks if block.get("type") == "tool_result"]

        if role == "assistant" and tool_uses:
            messages.append(
                {
                    "role": "assistant",
                    "content": "\n".join(text_parts),
                    "tool_calls": [
                        {
                            "id": str(tool_use.get("id") or f"call_{index}"),
                            "type": "function",
                            "function": {
                                "name": str(tool_use.get("name") or ""),
                                "arguments": json.dumps(tool_use.get("input") or {}),
                            },
                        }
                        for index, tool_use in enumerate(tool_uses)
                    ],
                }
            )
            continue

        if tool_results:
            for result in tool_results:
                result_content = result.get("content", "")
                if isinstance(result_content, list):
                    result_content = "\n".join(
                        str(block.get("text") or "")
                        for block in result_content
                        if isinstance(block, dict)
                    )
                elif not isinstance(result_content, str):
                    result_content = json.dumps(result_content)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(result.get("tool_use_id") or "call_0"),
                        "content": result_content,
                    }
                )
            visible_text = [text for text in text_parts if not text.strip().startswith("<system-reminder")]
            if visible_text:
                messages.append({"role": "user", "content": visible_text[-1].strip()})
            continue

        messages.append({"role": "assistant" if role == "assistant" else "user", "content": "\n".join(text_parts)})

    return messages


def anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    openai_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool["name"]),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema") or {},
                },
            }
        )
    return openai_tools


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": str(tool_choice["name"])}}
    return None


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    """Coerce to float and clamp to [lo, hi]; fall back to default if unusable.

    NVIDIA's GLM models validate sampling params server-side and return a hard
    HTTP 400 (e.g. "Temperature must be between 0 and 2, got 2.5") for anything
    out of range. Clients — especially OpenAI-style callers hitting
    /v1/chat/completions — forward whatever they want (temperature=2.5, top_p=2,
    frequency_penalty=3, ...), so we clamp before the request ever leaves the
    proxy instead of surfacing the 400.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf guard
        return default
    return lo if v < lo else hi if v > hi else v


def sanitize_openai_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a chat-completions payload for NVIDIA's GLM models.

    Two jobs:
    1. Clamp sampling params into the ranges NVIDIA accepts (out-of-range =
       HTTP 400 upstream). temperature [0,2], top_p [0,1], frequency_penalty
       and presence_penalty [-2,2].
    2. Drop fields GLM-5.2 doesn't support. The OpenAI path forwards the entire
       client body via dict(body), so unknown OpenAI-only fields (top_k, seed,
       n, logprobs, logit_bias, user, reasoning_effort, max_completion_tokens,
       response_format, stream_options, service_tier, ...) ride along and can
       either 400 or silently distort behavior. Keep an explicit allowlist so
       the upstream payload is exactly what GLM expects.
    """
    # ── max_tokens: positive int, default 1024 ────────────────────────────
    max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        # max_completion_tokens is the newer OpenAI spelling; honor it if the
        # caller used it instead of max_tokens.
        alt = payload.get("max_completion_tokens")
        if isinstance(alt, int) and alt > 0:
            max_tokens = alt
        else:
            max_tokens = 1024

    out: dict[str, Any] = {
        "model": str(payload.get("model") or ""),
        "messages": payload.get("messages") or [],
        "max_tokens": max_tokens,
        # temperature default 0.7 (the previous behavior); clamped to [0, 2].
        "temperature": _clamp(payload.get("temperature"), 0.0, 2.0, 0.7),
        "stream": bool(payload.get("stream", False)),
    }

    # top_p: optional, [0, 1]. Only send it if the caller supplied one —
    # GLM-5.2 defaults sensibly when absent.
    if payload.get("top_p") is not None:
        out["top_p"] = _clamp(payload.get("top_p"), 0.0, 1.0, 1.0)

    # penalties: optional, [-2, 2].
    for field in ("frequency_penalty", "presence_penalty"):
        if payload.get(field) is not None:
            out[field] = _clamp(payload.get(field), -2.0, 2.0, 0.0)

    # stop: passthrough but normalize to list[str] / drop if empty.
    stop = payload.get("stop")
    if isinstance(stop, str) and stop:
        out["stop"] = [stop]
    elif isinstance(stop, list) and stop:
        filtered = [s for s in stop if isinstance(s, str) and s]
        if filtered:
            out["stop"] = filtered

    # tools / tool_choice: passthrough (already OpenAI-shaped upstream of this).
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        out["tools"] = tools
        tc = payload.get("tool_choice")
        if tc is not None:
            out["tool_choice"] = tc
        else:
            out["tool_choice"] = "auto"

    # stream_options: when streaming, ask the upstream to emit a usage chunk on
    # the final event. Without this, NVIDIA's GLM endpoint never sends usage on
    # the stream path, so the proxy's _on_done / stats always see in_tokens=0
    # out_tokens=0 (the "in_tokens=0 on streams" cosmetic bug). include_usage is
    # standard OpenAI and harmless on non-streaming requests, but we only set it
    # when actually streaming to keep the payload minimal.
    if out.get("stream"):
        out["stream_options"] = {"include_usage": True}

    return out


def anthropic_openai_payload(body: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": anthropic_messages_to_openai(body),
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature", 0.7),
        # Honor the caller's stream flag so the upstream request actually
        # streams when the client asked for streaming. /v1/messages previously
        # ignored this and always buffered via handle_non_stream.
        "stream": bool(body.get("stream", False)),
    }
    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = anthropic_tool_choice_to_openai(body.get("tool_choice")) or "auto"
    # Sanitize (clamp sampling params, drop unsupported fields) before the
    # payload hits NVIDIA. Anthropic clients can still send out-of-range
    # temperature; clamp it instead of letting NVIDIA 400 the request.
    return sanitize_openai_payload(payload)


def openai_response_to_anthropic(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices") or []
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    content_blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        content_blocks.append({"type": "text", "text": content})

    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            tool_input = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": str(tool_call.get("id") or f"call_{len(content_blocks)}"),
                "name": str(function.get("name") or ""),
                "input": tool_input if isinstance(tool_input, dict) else {},
            }
        )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": extract_router_content(payload)})

    finish_reason = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
    stop_reason = _anthropic_stop_reason(finish_reason)
    response = anthropic_response_from_blocks(model, content_blocks, stop_reason)
    usage = payload.get("usage") or {}
    response["usage"] = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }
    return response


async def anthropic_sse_from_response(response: dict[str, Any]) -> AsyncIterator[bytes]:
    message = {**response, "content": []}
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': message}, separators=(',', ':'))}\n\n".encode()

    for index, block in enumerate(response.get("content") or []):
        start = {"type": "content_block_start", "index": index, "content_block": block}
        if block.get("type") == "text":
            start["content_block"] = {"type": "text", "text": ""}
        elif block.get("type") == "tool_use":
            start["content_block"] = {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": {},
            }
        yield f"event: content_block_start\ndata: {json.dumps(start, separators=(',', ':'))}\n\n".encode()

        if block.get("type") == "text" and block.get("text"):
            delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": block["text"]},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta, separators=(',', ':'))}\n\n".encode()
        elif block.get("type") == "tool_use":
            delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta, separators=(',', ':'))}\n\n".encode()

        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': index}, separators=(',', ':'))}\n\n".encode()

    delta = {
        "type": "message_delta",
        "delta": {"stop_reason": response.get("stop_reason") or "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": response.get("usage", {}).get("output_tokens", 0)},
    }
    yield f"event: message_delta\ndata: {json.dumps(delta, separators=(',', ':'))}\n\n".encode()
    yield b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"


async def anthropic_sse(model: str, content: str) -> AsyncIterator[bytes]:
    async for chunk in anthropic_sse_from_response(anthropic_response(model, content)):
        yield chunk


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    """Encode one Anthropic SSE event as bytes: 'event: <e>\\ndata: <json>\\n\\n'."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _anthropic_stop_reason(finish_reason: str | None) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    return {
        "tool_calls": "tool_use",
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }.get(finish_reason or "", "end_turn")


async def openai_sse_to_anthropic_sse(
    iterator: AsyncIterator[bytes],
    model: str,
    on_done: Any = None,
) -> AsyncIterator[bytes]:
    """Translate an OpenAI/NVIDIA SSE byte stream into Anthropic SSE events.

    Consumes raw bytes from NvidiaClient.stream_chat() and yields Anthropic
    event bytes as they arrive — message_start, content_block_start/delta/stop,
    message_delta, message_stop. Real streaming, no buffering of the full body.

    `on_done(prompt_tokens, completion_tokens, total_tokens, tool_calls)` is
    invoked once with the final usage if the upstream reports it; the caller
    wires this to stats. Optional.
    """
    message_id = f"msg_{uuid.uuid4().hex}"
    started = False
    # Content-block bookkeeping. Anthropic blocks are indexed in emission
    # order: a text block (if any) at index 0, then tool_use blocks after.
    text_block_open = False
    # openai tool_call.index -> anthropic block index
    tool_block_index: dict[int, int] = {}
    next_block_index = 0  # next anthropic block index to hand out
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    tool_calls_seen = 0
    buffer = b""

    def _ensure_started() -> bytes | None:
        nonlocal started
        if started:
            return None
        started = True
        message = {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return _sse_event("message_start", {"type": "message_start", "message": message})

    def _close_all_blocks() -> list[bytes]:
        """Emit content_block_stop for every currently-open block."""
        out: list[bytes] = []
        # Text block is index 0; tool blocks are the rest, in order.
        indices = sorted(tool_block_index.values())
        if text_block_open:
            indices = [0] + indices
        for idx in indices:
            out.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": idx}))
        return out

    async def _flush_final() -> AsyncIterator[bytes]:
        # Close any open blocks, then message_delta + message_stop.
        for chunk in _close_all_blocks():
            yield chunk
        delta = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
        yield _sse_event("message_delta", delta)
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        if on_done is not None:
            try:
                on_done(input_tokens, output_tokens, total_tokens, tool_calls_seen)
            except Exception:
                pass

    # Walk the byte stream, buffering partial SSE lines.
    async for raw in iterator:
        buffer += raw
        # SSE events are separated by blank lines; process complete lines.
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            data_str = line[5:].strip()
            if data_str == b"[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                continue

            # Upstream emitted a terminal error chunk (e.g. mid-stream read
            # timeout from NvidiaClient). Surface it as a text block so the
            # client sees the failure instead of an empty message.
            if isinstance(chunk, dict) and isinstance(chunk.get("error"), dict):
                start_evt = _ensure_started()
                if start_evt is not None:
                    yield start_evt
                if not text_block_open:
                    text_block_open = True
                    next_block_index = 1
                    yield _sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    )
                err = chunk["error"]
                err_text = str(err.get("message") or "upstream stream error")
                rid = err.get("rid")
                tag = f"[stream error rid={rid}] {err_text}" if rid else f"[stream error] {err_text}"
                yield _sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": tag}},
                )
                continue

            # Adopt the upstream message id once we see it.
            if not started and isinstance(chunk.get("id"), str):
                message_id = chunk["id"]

            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            choice = choices[0] if isinstance(choices, list) and choices else {}
            if not isinstance(choice, dict):
                choice = {}
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

            # ── text delta ──────────────────────────────────────────────
            content = delta.get("content")
            if isinstance(content, str) and content:
                start_evt = _ensure_started()
                if start_evt is not None:
                    yield start_evt
                if not text_block_open:
                    text_block_open = True
                    next_block_index = 1  # text is index 0; tools come after
                    yield _sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    )
                yield _sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": content}},
                )

            # ── tool-call delta ─────────────────────────────────────────
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                start_evt = _ensure_started()
                if start_evt is not None:
                    yield start_evt
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    oai_index = int(tc.get("index") or 0)
                    if oai_index not in tool_block_index:
                        tool_block_index[oai_index] = next_block_index
                        next_block_index += 1
                        tool_calls_seen += 1
                        function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        yield _sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": tool_block_index[oai_index],
                                "content_block": {
                                    "type": "tool_use",
                                    "id": str(tc.get("id") or f"call_{oai_index}"),
                                    "name": str(function.get("name") or ""),
                                    "input": {},
                                },
                            },
                        )
                    function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    args_fragment = function.get("arguments")
                    if args_fragment:
                        yield _sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": tool_block_index[oai_index],
                                "delta": {"type": "input_json_delta", "partial_json": str(args_fragment)},
                            },
                        )

            # ── finish_reason ───────────────────────────────────────────
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                stop_reason = _anthropic_stop_reason(finish_reason)

            # ── usage (often on the final chunk) ────────────────────────
            usage = chunk.get("usage") if isinstance(chunk, dict) else None
            if isinstance(usage, dict):
                input_tokens = int(usage.get("prompt_tokens") or 0)
                output_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or 0)

    # Stream ended. If we never started (empty upstream), still emit a minimal
    # valid Anthropic stream so the client gets a well-formed empty message.
    if not started:
        start_evt = _ensure_started()
        if start_evt is not None:
            yield start_evt

    async for chunk in _flush_final():
        yield chunk

