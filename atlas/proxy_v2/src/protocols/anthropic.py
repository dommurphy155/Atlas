"""Anthropic protocol adapter."""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from src.core.types import (
    BlockType,
    ContentBlock,
    FinishReason,
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
    """Generate a message ID."""
    return f"msg_{uuid.uuid4().hex}"


# ============================================================================
# Request Parsing
# ============================================================================


def parse_anthropic_request(data: dict[str, Any]) -> Request:
    """Parse Anthropic request to internal format."""

    # Parse messages
    messages = [parse_message(msg) for msg in data.get("messages", [])]

    # Parse system prompt
    system = parse_system(data.get("system"))

    # Parse options
    options = parse_options(data)

    # Determine streaming
    stream = data.get("stream", False)

    # Get model
    model = data.get("model", "unknown")

    return Request(
        model=model,
        messages=messages,
        system=system,
        options=options,
        stream=stream,
    )


def parse_system(system: Any) -> Optional[str]:
    """Parse Anthropic system field."""
    if not system:
        return None

    if isinstance(system, str):
        return system

    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        return "\n".join(parts) if parts else None

    return str(system)


def parse_message(msg: dict[str, Any]) -> Message:
    """Parse Anthropic message to internal format."""
    role = msg.get("role", "user")
    content = msg.get("content", "")

    # Handle role:system messages
    if role == "system":
        text = parse_content_as_text(content)
        return Message(role="system", content=text)

    # Parse content blocks
    content_blocks: list[ContentBlock] = []
    tool_calls: list[ToolCall] = []

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "text")

            if block_type == "text":
                text = block.get("text", "")
                if text:
                    content_blocks.append(ContentBlock.text(text))

            elif block_type == "thinking":
                thinking = block.get("thinking", "")
                if thinking:
                    content_blocks.append(ContentBlock.thinking(thinking))

            elif block_type == "tool_use":
                tc = ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                )
                tool_calls.append(tc)
                content_blocks.append(ContentBlock.tool_use(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                ))

            elif block_type == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    text_parts = []
                    for b in result_content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            text_parts.append(b.get("text", ""))
                    result_content = "\n".join(text_parts)
                content_blocks.append(ContentBlock.tool_result(
                    tool_use_id=block.get("tool_use_id", ""),
                    content=str(result_content),
                    is_error=block.get("is_error", False),
                ))

            elif block_type == "image":
                # Handle image blocks
                source = block.get("source", {})
                if source:
                    from src.core.types import ImageSource
                    content_blocks.append(ContentBlock.image(ImageSource(
                        type=source.get("type", "base64"),
                        media_type=source.get("media_type", "image/png"),
                        data=source.get("data", ""),
                        url=source.get("url", ""),
                    )))

    elif isinstance(content, str):
        if content:
            content_blocks.append(ContentBlock.text(content))

    # Determine effective role for message
    effective_role = role
    if role == "assistant" and tool_calls:
        effective_role = "assistant"
    elif role == "tool":
        effective_role = "tool"

    return Message(
        role=effective_role,
        content=content_blocks[0].text if content_blocks and content_blocks[0].type == BlockType.TEXT else None,
        content_blocks=content_blocks,
        tool_calls=tool_calls,
        tool_call_id=msg.get("tool_call_id"),
    )


def parse_content_as_text(content: Any) -> Optional[str]:
    """Parse content to plain text."""
    if not content:
        return None

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        return "\n".join(parts) if parts else None

    return str(content)


def parse_options(data: dict[str, Any]) -> RequestOptions:
    """Parse Anthropic options to internal format."""

    # Parse tools
    tools: list[ToolDefinition] = []
    for tool in data.get("tools", []):
        if isinstance(tool, dict):
            tools.append(ToolDefinition(
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                parameters=tool.get("input_schema", {}),
            ))

    # Parse tool choice
    tool_choice = None
    if "tool_choice" in data:
        tool_choice_data = data.get("tool_choice")
        if isinstance(tool_choice_data, dict):
            tc_type = tool_choice_data.get("type", "auto")
            if tc_type == "any":
                tc_type = "required"
            tool_choice = ToolChoice(
                type=tc_type,
                name=tool_choice_data.get("name"),
                disable_parallel_tool_use=tool_choice_data.get("disable_parallel_tool_use", False),
            )

    # Parse thinking config
    thinking = None
    if "thinking" in data:
        thinking_config = data.get("thinking", {})
        if isinstance(thinking_config, dict):
            if thinking_config.get("type") == "enabled":
                thinking = {
                    "type": "enabled",
                    "budget_tokens": thinking_config.get("budget_tokens", 4096),
                }

    return RequestOptions(
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 4096),
        top_p=data.get("top_p"),
        tools=tools if tools else None,
        tool_choice=tool_choice,
        thinking=thinking,
    )


# ============================================================================
# Response Formatting
# ============================================================================


def format_anthropic_response(response: Response) -> dict[str, Any]:
    """Format internal response to Anthropic format."""

    # Build content blocks
    content_blocks: list[dict[str, Any]] = []

    # Add thinking first
    if response.thinking:
        content_blocks.append({
            "type": "thinking",
            "thinking": response.thinking,
        })

    # Add text content
    if response.content:
        content_blocks.append({
            "type": "text",
            "text": response.content,
        })

    # Add tool uses
    for tc in response.tool_calls:
        content_blocks.append({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": tc.arguments,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    # Map finish reason
    stop_reason = map_finish_reason(response.finish_reason, bool(response.tool_calls))

    return {
        "id": response.id,
        "type": "message",
        "role": "assistant",
        "model": response.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    }


def map_finish_reason(finish_reason: FinishReason, has_tool_calls: bool) -> str:
    """Map internal finish reason to Anthropic stop_reason."""
    if has_tool_calls:
        return "tool_use"

    mapping = {
        FinishReason.STOP: "end_turn",
        FinishReason.LENGTH: "max_tokens",
        FinishReason.CONTENT_FILTER: "end_turn",
    }
    return mapping.get(finish_reason, "end_turn")


def format_anthropic_error(error: Exception) -> dict[str, Any]:
    """Format error to Anthropic format."""
    from src.core.errors import APIError, ProxyError

    if isinstance(error, APIError):
        return error.to_anthropic_dict()

    # Generic error
    return {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": str(error),
        },
    }


# ============================================================================
# SSE Event Formatting
# ============================================================================


def format_anthropic_stream_event(response: Response, event_type: str) -> bytes:
    """Format a streaming event for Anthropic."""
    data: dict[str, Any]

    if event_type == "message_start":
        data = {
            "type": "message_start",
            "message": {
                "id": response.id,
                "type": "message",
                "role": "assistant",
                "model": response.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": 0,
                },
            },
        }

    elif event_type == "content_block_start":
        # Determine block type
        if response.thinking:
            data = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            }
        elif response.tool_calls:
            tc = response.tool_calls[0]
            data = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": {},
                },
            }
        else:
            data = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }

    elif event_type == "content_block_delta":
        if response.thinking:
            data = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": response.thinking},
            }
        elif response.tool_calls:
            tc = response.tool_calls[0]
            data = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(tc.arguments)},
            }
        else:
            data = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": response.content},
            }

    elif event_type == "content_block_stop":
        data = {"type": "content_block_stop", "index": 0}

    elif event_type == "message_delta":
        stop_reason = map_finish_reason(response.finish_reason, bool(response.tool_calls))
        data = {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
            },
            "usage": {"output_tokens": response.usage.completion_tokens},
        }

    elif event_type == "message_stop":
        data = {"type": "message_stop"}

    else:
        data = {"type": event_type}

    return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()
