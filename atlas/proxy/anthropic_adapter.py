"""Anthropic protocol adapter for Atlas.

This module handles Anthropic /v1/messages endpoint conversion,
preserving full protocol fidelity including:
- Content block preservation (text, tool_use, tool_result, thinking)
- Proper SSE event sequencing
- Tool use/result round-tripping

Key design inspired by Ollama's StreamConverter:
- State-based conversion with proper block indexing
- Thinking delta support
- Input token estimation
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from proxy.internal import (
    ContentBlock,
    Message,
    Request,
    Response,
    ToolCall,
    ToolDefinition,
    Usage,
)


def anthropic_to_internal(body: dict[str, Any], upstream_model: str) -> Request:
    """Convert Anthropic MessagesRequest to InternalRequest.

    Preserves all content blocks including thinking, tool_use, tool_result.
    """
    # Parse system prompt
    system = _parse_system(body.get("system"))

    # Parse messages
    messages = _parse_messages(body.get("messages", []))

    # Parse tools
    tools = _parse_tools(body.get("tools"))

    # Parse options
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 1024)
    top_p = body.get("top_p")
    stream = bool(body.get("stream", False))
    thinking = body.get("thinking")

    return Request(
        model=upstream_model,
        messages=messages,
        system=system,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=stream,
        thinking=thinking,
    )


def _parse_system(system: Any) -> str | None:
    """Parse Anthropic system field (string or list of text blocks)."""
    if not system:
        return None

    if isinstance(system, str):
        return system

    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts) if parts else None

    return str(system)


def _parse_messages(raw_messages: list[dict[str, Any]]) -> list[Message]:
    """Parse Anthropic messages to InternalMessage list."""
    messages: list[Message] = []

    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role", "user"))
        content = msg.get("content", "")

        # Handle role:system messages in the messages array
        if role == "system":
            sys_text = _parse_content_as_text(content)
            if sys_text:
                messages.append(Message(role="system", content=sys_text))
            continue

        # Parse content blocks
        blocks: list[ContentBlock] = []
        tool_calls: list[ToolCall] = []

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type", "text")

                if block_type == "text":
                    text = str(block.get("text", ""))
                    if text:
                        blocks.append(ContentBlock.from_text(text))

                elif block_type == "thinking":
                    thinking = str(block.get("thinking", ""))
                    if thinking:
                        blocks.append(ContentBlock.from_thinking(thinking))

                elif block_type == "tool_use":
                    tc = ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                    tool_calls.append(tc)

                elif block_type == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        # Nested content blocks
                        text_parts = []
                        for b in result_content:
                            if isinstance(b, dict) and b.get("type") == "text":
                                text_parts.append(str(b.get("text", "")))
                        result_content = "\n".join(text_parts)
                    blocks.append(ContentBlock.from_tool_result(
                        tool_use_id=block.get("tool_use_id", ""),
                        content=str(result_content),
                    ))
        else:
            # Simple string content
            text = _parse_content_as_text(content)
            if text:
                blocks.append(ContentBlock.from_text(text))

        # Create message
        if role == "assistant":
            msg_obj = Message(
                role="assistant",
                content=blocks if blocks else None,
                tool_calls=tool_calls if tool_calls else [],
            )
        elif role == "tool":
            # Tool results - get tool_call_id from the first tool call if available
            tool_call_id = None
            if blocks and blocks[0].tool_use_id:
                tool_call_id = blocks[0].tool_use_id
            msg_obj = Message(
                role="tool",
                content=blocks[0].content if blocks else "",
                tool_call_id=tool_call_id or msg.get("tool_call_id"),
            )
        else:
            msg_obj = Message(
                role=role,
                content=blocks if blocks else None,
            )

        messages.append(msg_obj)

    return messages


def _parse_content_as_text(content: Any) -> str | None:
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
                    parts.append(str(block.get("text", "")))
        return "\n".join(parts) if parts else None

    return str(content)


def _parse_tools(tools: Any) -> list[ToolDefinition] | None:
    """Parse Anthropic tools to InternalToolDefinition list."""
    if not isinstance(tools, list) or not tools:
        return None

    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        name = tool.get("name")
        if not name:
            continue

        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {})

        result.append(ToolDefinition(
            name=name,
            description=description,
            parameters=input_schema,
        ))

    return result if result else None


# ============================================================================
# StreamConverter - State-based SSE conversion (inspired by Ollama)
# ============================================================================

@dataclass
class StreamState:
    """State for converting OpenAI stream to Anthropic SSE."""
    message_id: str
    model: str
    first_write: bool = True
    content_index: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_input_tokens: int = 0

    # Block state tracking
    thinking_started: bool = False
    thinking_done: bool = False
    text_started: bool = False
    tool_calls_sent: dict[str, bool] = field(default_factory=dict)


class StreamConverter:
    """Stateful converter for OpenAI SSE to Anthropic SSE.

    Similar to Ollama's StreamConverter, this maintains state across
    streaming chunks to properly sequence events.
    """

    def __init__(self, model: str, estimated_input_tokens: int = 0):
        self.state = StreamState(
            message_id=f"msg_{uuid.uuid4().hex}",
            model=model,
            estimated_input_tokens=estimated_input_tokens,
        )

    def process_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Process a single OpenAI chunk and return Anthropic events."""
        events = []
        state = self.state

        # First write: emit message_start
        if state.first_write:
            state.first_write = False
            # Use actual metrics if available, otherwise use estimate
            usage = chunk.get("usage")
            if usage is not None:
                state.input_tokens = usage.get("prompt_tokens", 0) if state.input_tokens == 0 else state.input_tokens
            if state.input_tokens == 0 and state.estimated_input_tokens > 0:
                state.input_tokens = state.estimated_input_tokens

            events.append({
                "event": "message_start",
                "data": {
                    "type": "message_start",
                    "message": {
                        "id": state.message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": state.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": state.input_tokens,
                            "output_tokens": 0,
                        },
                    },
                },
            })

        # Get delta content
        choices = chunk.get("choices", [])
        if not choices:
            return events

        delta = choices[0].get("delta", {})
        if not delta:
            return events

        # Handle thinking delta (if backend supports it)
        thinking = delta.get("thinking")
        if thinking:
            # Close text block if open
            if state.text_started and not state.thinking_done:
                events.append(self._content_block_stop())
                state.content_index += 1
                state.text_started = False

            # Start thinking block if not started
            if not state.thinking_started:
                state.thinking_started = True
                events.append({
                    "event": "content_block_start",
                    "data": {
                        "type": "content_block_start",
                        "index": state.content_index,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "",
                        },
                    },
                })

            # Emit thinking delta
            events.append({
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": state.content_index,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": thinking,
                    },
                },
            })

        # Handle text delta
        content = delta.get("content")
        if content:
            # Close thinking block if open
            if state.thinking_started and not state.thinking_done:
                state.thinking_done = True
                events.append(self._content_block_stop())
                state.content_index += 1

            # Start text block if not started
            if not state.text_started:
                state.text_started = True
                events.append({
                    "event": "content_block_start",
                    "data": {
                        "type": "content_block_start",
                        "index": state.content_index,
                        "content_block": {
                            "type": "text",
                            "text": "",
                        },
                    },
                })

            # Emit text delta
            events.append({
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": state.content_index,
                    "delta": {
                        "type": "text_delta",
                        "text": content,
                    },
                },
            })

        # Handle tool calls
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            tc_id = tc.get("id")
            if not tc_id or state.tool_calls_sent.get(tc_id):
                continue

            state.tool_calls_sent[tc_id] = True

            # Close open blocks before starting tool
            if state.thinking_started and not state.thinking_done:
                events.append(self._content_block_stop())
                state.content_index += 1
                state.thinking_done = True

            if state.text_started:
                events.append(self._content_block_stop())
                state.content_index += 1
                state.text_started = False

            func = tc.get("function", {})
            events.append({
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": state.content_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc_id,
                        "name": func.get("name", ""),
                        "input": {},
                    },
                },
            })

            # Handle arguments
            args = func.get("arguments", "")
            if args:
                events.append({
                    "event": "content_block_delta",
                    "data": {
                        "type": "content_block_delta",
                        "index": state.content_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": str(args),
                        },
                    },
                })

        # Handle final chunk (done)
        finish_reason = choices[0].get("finish_reason")
        if finish_reason:
            # Close any open blocks
            if state.thinking_started and not state.thinking_done:
                events.append(self._content_block_stop())
                state.content_index += 1

            if state.text_started:
                events.append(self._content_block_stop())
                state.content_index += 1

            # Update tokens from usage
            usage = chunk.get("usage")
            if usage is not None:
                state.input_tokens = usage.get("prompt_tokens", state.input_tokens)
                state.output_tokens = usage.get("completion_tokens", 0)

            stop_reason = _map_stop_reason(finish_reason, len(state.tool_calls_sent) > 0)

            events.append({
                "event": "message_delta",
                "data": {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": None,
                    },
                    "usage": {
                        "output_tokens": state.output_tokens,
                    },
                },
            })

            events.append({
                "event": "message_stop",
                "data": {"type": "message_stop"},
            })

        return events

    def _content_block_stop(self) -> dict[str, Any]:
        return {
            "event": "content_block_stop",
            "data": {
                "type": "content_block_stop",
                "index": self.state.content_index,
            },
        }

    @property
    def message_id(self) -> str:
        return self.state.message_id


def _map_stop_reason(finish_reason: str | None, has_tool_calls: bool) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    if has_tool_calls:
        return "tool_use"

    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason or "", "end_turn")


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    """Encode one Anthropic SSE event as bytes."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


# ============================================================================
# Streaming adapter
# ============================================================================

async def convert_openai_stream_to_anthropic(
    iterator: AsyncIterator[bytes],
    model: str,
    on_done: Any = None,
    estimated_input_tokens: int = 0,
) -> AsyncIterator[bytes]:
    """Convert OpenAI/NVIDIA SSE stream to Anthropic SSE events.

    Uses stateful StreamConverter for proper event sequencing.
    """
    converter = StreamConverter(model, estimated_input_tokens)
    buffer = b""

    async for raw in iterator:
        buffer += raw

        # Process complete lines
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

            # Handle error chunks
            if isinstance(chunk, dict) and isinstance(chunk.get("error"), dict):
                # Emit error as text block
                err = chunk["error"]
                err_text = str(err.get("message") or "upstream stream error")
                events = [{
                    "event": "content_block_start",
                    "data": {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                }, {
                    "event": "content_block_delta",
                    "data": {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": err_text},
                    },
                }, {
                    "event": "content_block_stop",
                    "data": {
                        "type": "content_block_stop",
                        "index": 0,
                    },
                }]

                for evt in events:
                    yield _sse_event(evt["event"], evt["data"])
                continue

            # Process normal chunk
            events = converter.process_chunk(chunk)
            for evt in events:
                yield _sse_event(evt["event"], evt["data"])

    # Call on_done if provided
    if on_done:
        try:
            on_done(
                converter.state.input_tokens,
                converter.state.output_tokens,
                converter.state.input_tokens + converter.state.output_tokens,
                len(converter.state.tool_calls_sent),
            )
        except Exception:
            pass


async def generate_anthropic_sse(response: Response) -> AsyncIterator[bytes]:
    """Generate Anthropic SSE events from InternalResponse (non-streaming)."""
    message_id = f"msg_{uuid.uuid4().hex}"

    # Build content blocks
    blocks: list[dict[str, Any]] = []

    # Add thinking if present
    if response.thinking:
        blocks.append({"type": "thinking", "thinking": response.thinking})

    # Add text content
    if response.content:
        blocks.append({"type": "text", "text": response.content})

    # Add tool uses
    for tc in response.tool_calls:
        blocks.append({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": tc.arguments,
        })

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    # message_start
    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": "",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    # Process each block
    for idx, block in enumerate(blocks):
        block_type = block.get("type", "text")

        # content_block_start
        start_block = _make_block_start(block_type, block)
        yield _sse_event("content_block_start", {
            "type": "content_block_start",
            "index": idx,
            "content_block": start_block,
        })

        # content_block_delta
        delta_data = _make_block_delta(block_type, block)
        if delta_data:
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": idx,
                "delta": delta_data,
            })

        # content_block_stop
        yield _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": idx,
        })

    # Map stop reason
    stop_reason = _map_stop_reason(response.stop_reason, len(response.tool_calls) > 0)

    # message_delta
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": None,
        },
        "usage": {"output_tokens": response.usage.completion_tokens},
    })

    # message_stop
    yield _sse_event("message_stop", {"type": "message_stop"})


def _make_block_start(block_type: str, block: dict[str, Any]) -> dict[str, Any]:
    """Create content_block_start content for a block."""
    if block_type == "text":
        return {"type": "text", "text": ""}
    elif block_type == "thinking":
        return {"type": "thinking", "thinking": ""}
    elif block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "input": {},
        }
    return {"type": "text", "text": ""}


def _make_block_delta(block_type: str, block: dict[str, Any]) -> dict[str, Any] | None:
    """Create content_block_delta delta for a block."""
    if block_type == "text":
        text = block.get("text", "")
        if text:
            return {"type": "text_delta", "text": text}
    elif block_type == "thinking":
        thinking = block.get("thinking", "")
        if thinking:
            return {"type": "thinking_delta", "thinking": thinking}
    elif block_type == "tool_use":
        input_data = block.get("input", {})
        if input_data:
            return {"type": "input_json_delta", "partial_json": json.dumps(input_data)}
    return None
