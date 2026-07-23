"""OpenAI protocol adapter."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from src.core.types import (
    BlockType,
    Capability,
    ContentBlock,
    FinishReason,
    ImageSource,
    Logprobs,
    Message,
    Request,
    RequestOptions,
    Response,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    Usage,
)


def generate_id() -> str:
    """Generate a completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex}"


# ============================================================================
# Request Parsing
# ============================================================================


def parse_openai_request(data: dict[str, Any]) -> Request:
    """Parse OpenAI request to internal format."""

    # Parse messages
    messages = [parse_message(msg) for msg in data.get("messages", [])]

    # Parse options
    options = parse_options(data)

    # Determine streaming
    stream = data.get("stream", False)

    # Get model
    model = data.get("model", "unknown")

    # Get system prompt
    system = extract_system_prompt(messages)

    return Request(
        model=model,
        messages=messages,
        system=system,
        options=options,
        stream=stream,
    )


def parse_message(msg: dict[str, Any]) -> Message:
    """Parse OpenAI message to internal format."""
    role = msg.get("role", "user")
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls", [])
    tool_call_id = msg.get("tool_call_id")
    name = msg.get("name")

    # Parse content blocks
    content_blocks: list[ContentBlock] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "text")
                if block_type == "text":
                    content_blocks.append(ContentBlock.text(block.get("text", "")))
                elif block_type == "image_url":
                    # Handle image URLs
                    url_data = block.get("image_url", {})
                    if isinstance(url_data, dict):
                        url = url_data.get("url", "")
                        if url.startswith("data:"):
                            # Base64 image
                            import base64
                            parts = url.split(",", 1)
                            if len(parts) == 2:
                                media_type = parts[0].replace("data:", "").replace(";base64", "")
                                data = parts[1]
                                content_blocks.append(ContentBlock.image(
                                    ImageSource(
                                        type="base64",
                                        media_type=media_type,
                                        data=data,
                                    )
                                ))
                elif block_type == "tool_use":
                    content_blocks.append(ContentBlock.tool_use(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                    ))
                elif block_type == "tool_result":
                    content_blocks.append(ContentBlock.tool_result(
                        tool_use_id=block.get("tool_use_id", ""),
                        content=str(block.get("content", "")),
                    ))
    elif isinstance(content, str):
        if content:
            content_blocks.append(ContentBlock.text(content))

    # Parse tool calls
    parsed_tool_calls: list[ToolCall] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            parsed_tool_calls.append(ToolCall.from_dict(tc))

    return Message(
        role=role,
        content=content if isinstance(content, str) else None,
        content_blocks=content_blocks,
        tool_calls=parsed_tool_calls,
        tool_call_id=tool_call_id,
        name=name,
    )


def extract_system_prompt(messages: list[Message]) -> Optional[str]:
    """Extract system prompt from messages."""
    for msg in messages:
        if msg.role == "system":
            return msg.content
    return None


def parse_options(data: dict[str, Any]) -> RequestOptions:
    """Parse OpenAI options to internal format."""

    # Parse tools
    tools: list[ToolDefinition] = []
    for tool in data.get("tools", []):
        if isinstance(tool, dict):
            tools.append(ToolDefinition.from_dict(tool))

    # Parse tool choice
    tool_choice = None
    if "tool_choice" in data:
        tool_choice = ToolChoice.from_dict(data.get("tool_choice"))

    # Parse thinking config (for reasoning models)
    thinking = None
    if "thinking" in data or "reasoning_effort" in data:
        reasoning_effort = data.get("thinking", {}).get("effort") or data.get("reasoning_effort")
        if reasoning_effort:
            thinking = {"type": "enabled", "budget_tokens": data.get("thinking", {}).get("budget_tokens", 4096)}

    return RequestOptions(
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 4096),
        top_p=data.get("top_p"),
        top_k=data.get("top_k"),
        stop=data.get("stop"),
        frequency_penalty=data.get("frequency_penalty", 0.0),
        presence_penalty=data.get("presence_penalty", 0.0),
        seed=data.get("seed"),
        response_format=data.get("response_format"),
        tools=tools if tools else None,
        tool_choice=tool_choice,
        thinking=thinking,
        logprobs=data.get("logprobs", False),
        top_logprobs=data.get("top_logprobs", 0),
    )


# ============================================================================
# Response Formatting
# ============================================================================


def format_openai_response(response: Response) -> dict[str, Any]:
    """Format internal response to OpenAI format."""

    # Determine finish reason
    finish_reason = response.finish_reason.value
    if response.tool_calls:
        finish_reason = "tool_calls"

    # Build message
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
    }

    if response.tool_calls:
        message["tool_calls"] = [tc.to_dict() for tc in response.tool_calls]

    return {
        "id": response.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": response.usage.to_dict(),
    }


def format_openai_stream_chunk(response: Response) -> bytes:
    """Format internal response to OpenAI streaming format."""

    # Determine finish reason
    finish_reason = response.finish_reason.value
    if response.tool_calls:
        finish_reason = "tool_calls"

    # Build delta
    delta: dict[str, Any] = {
        "content": response.content,
    }

    if response.tool_calls:
        delta["tool_calls"] = [tc.to_dict() for tc in response.tool_calls]

    if response.thinking:
        delta["thinking"] = response.thinking

    chunk = {
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason if response.finish_reason != FinishReason.STOP else None,
            }
        ],
    }

    # Add usage to final chunk
    if response.finish_reason != FinishReason.STOP:
        chunk["usage"] = response.usage.to_dict()

    return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()


def format_openai_error(error: Exception) -> dict[str, Any]:
    """Format error to OpenAI format."""
    from src.core.errors import APIError, ProxyError

    if isinstance(error, APIError):
        return error.to_openai_dict()

    # Generic error
    return {
        "error": {
            "message": str(error),
            "type": "api_error",
            "code": 500,
        }
    }


# ============================================================================
# Stream Chunk Parsing
# ============================================================================


def parse_openai_stream_chunk(data: bytes) -> Optional[dict[str, Any]]:
    """Parse OpenAI streaming chunk."""
    line = data.strip()
    if not line.startswith(b"data:"):
        return None

    data_str = line[5:].strip()
    if data_str == b"[DONE]":
        return None

    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None
