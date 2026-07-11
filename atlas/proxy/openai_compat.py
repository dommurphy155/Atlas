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


def anthropic_openai_payload(body: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": anthropic_messages_to_openai(body),
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature", 0.7),
        "stream": False,
    }
    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = anthropic_tool_choice_to_openai(body.get("tool_choice")) or "auto"
    return payload


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
    stop_reason = "tool_use" if finish_reason == "tool_calls" or any(block["type"] == "tool_use" for block in content_blocks) else "end_turn"
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
