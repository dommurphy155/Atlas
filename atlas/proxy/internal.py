"""Internal canonical message format for Atlas.

This module defines the canonical representation that all protocol adapters
convert to/from. This allows the proxy to:
1. Support multiple protocols (OpenAI, Anthropic) without code duplication
2. Preserve all content blocks (thinking, tool_use, etc.)
3. Enable future backend abstraction

All adapters should convert to InternalRequest before processing,
and convert from InternalResponse for output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentBlock:
    """A single content block in a message.

    Supported types:
    - text: Plain text content
    - tool_use: A tool call from the model
    - tool_result: A result from a tool execution
    - thinking: Reasoning/thinking content (Anthropic)
    """
    type: str  # "text", "tool_use", "tool_result", "thinking"
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    tool_use_id: str | None = None
    content: str | None = None
    thinking: str | None = None

    @classmethod
    def from_text(cls, text: str) -> ContentBlock:
        return cls(type="text", text=text)

    @classmethod
    def from_tool_use(cls, id: str, name: str, input: dict[str, Any]) -> ContentBlock:
        return cls(type="tool_use", id=id, name=name, input=input)

    @classmethod
    def from_tool_result(cls, tool_use_id: str, content: str) -> ContentBlock:
        return cls(type="tool_result", tool_use_id=tool_use_id, content=content)

    @classmethod
    def from_thinking(cls, thinking: str) -> ContentBlock:
        return cls(type="thinking", thinking=thinking)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {"type": self.type}
        if self.text is not None:
            result["text"] = self.text
        if self.id is not None:
            result["id"] = self.id
        if self.name is not None:
            result["name"] = self.name
        if self.input is not None:
            result["input"] = self.input
        if self.tool_use_id is not None:
            result["tool_use_id"] = self.tool_use_id
        if self.content is not None:
            result["content"] = self.content
        if self.thinking is not None:
            result["thinking"] = self.thinking
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentBlock:
        """Create from dictionary format."""
        return cls(
            type=data.get("type", "text"),
            text=data.get("text"),
            id=data.get("id"),
            name=data.get("name"),
            input=data.get("input"),
            tool_use_id=data.get("tool_use_id"),
            content=data.get("content"),
            thinking=data.get("thinking"),
        )


@dataclass
class ToolCall:
    """A tool call (function invocation) from the model."""
    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        func = data.get("function", {})
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        return cls(
            id=data.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


@dataclass
class ToolDefinition:
    """Definition of a tool available to the model."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolDefinition:
        # Handle both OpenAI format (function) and Anthropic format (input_schema)
        func = data.get("function", {})
        return cls(
            name=func.get("name") or data.get("name", ""),
            description=func.get("description") or "",
            parameters=func.get("parameters") or data.get("input_schema", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str | list[ContentBlock] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def get_text_content(self) -> str:
        """Extract plain text from content."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        # It's a list of content blocks
        parts = []
        for block in self.content:
            if isinstance(block, ContentBlock):
                if block.text:
                    parts.append(block.text)
                elif block.thinking:
                    parts.append(block.thinking)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "thinking":
                    parts.append(str(block.get("thinking", "")))
        return "\n".join(parts)

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI message format."""
        result: dict[str, Any] = {"role": self.role}

        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        if self.name:
            result["name"] = self.name

        # Handle content
        if isinstance(self.content, list):
            # Convert ContentBlock list to OpenAI format
            text_parts = []
            for block in self.content:
                if isinstance(block, ContentBlock):
                    if block.type == "text":
                        text_parts.append({"type": "text", "text": block.text or ""})
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append({"type": "text", "text": block.get("text", "")})
            if text_parts:
                result["content"] = "\n".join(b["text"] for b in text_parts if b.get("text"))
            else:
                result["content"] = ""
        else:
            result["content"] = self.content or ""

        return result

    @classmethod
    def from_openai_dict(cls, data: dict[str, Any]) -> Message:
        """Create from OpenAI message format."""
        content = data.get("content", "")
        tool_calls = []
        for tc in data.get("tool_calls", []):
            tool_calls.append(ToolCall.from_dict(tc))

        return cls(
            role=data.get("role", "user"),
            content=content,
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass
class Request:
    """Canonical request representation."""
    model: str
    messages: list[Message]
    system: str | None = None
    tools: list[ToolDefinition] | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float | None = None
    stream: bool = False
    stop: list[str] | None = None
    # Anthropic-specific
    thinking: bool | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI chat completions format."""
        result: dict[str, Any] = {
            "model": self.model,
            "messages": [msg.to_openai_dict() for msg in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }

        if self.system:
            # Insert system message at the beginning
            result["messages"].insert(0, {"role": "system", "content": self.system})

        if self.tools:
            result["tools"] = [tool.to_dict() for tool in self.tools]

        if self.top_p is not None:
            result["top_p"] = self.top_p

        if self.stop:
            result["stop"] = self.stop

        return result


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Response:
    """Canonical response representation."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    # For streaming - content blocks being built up
    content_blocks: list[ContentBlock] = field(default_factory=list)
    # Thinking content if present
    thinking: str | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI chat completion format."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}

        if self.tool_calls:
            message["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        return {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if self.tool_calls else self.stop_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }

    def to_anthropic_message(self, model: str) -> dict[str, Any]:
        """Convert to Anthropic MessagesResponse format."""
        blocks: list[dict[str, Any]] = []

        # Add thinking if present
        if self.thinking:
            blocks.append({"type": "thinking", "thinking": self.thinking})

        # Add text content
        if self.content:
            blocks.append({"type": "text", "text": self.content})

        # Add tool uses
        for tc in self.tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })

        if not blocks:
            blocks.append({"type": "text", "text": ""})

        return {
            "id": f"msg_{json.dumps({})[1:9]}",  # Simple ID generation
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": blocks,
            "stop_reason": "tool_use" if self.tool_calls else self.stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": self.usage.prompt_tokens,
                "output_tokens": self.usage.completion_tokens,
            },
        }


# Helper functions for conversion

def messages_to_internal(
    role: str,
    content: str | list[dict[str, Any]] | None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
    name: str | None = None,
) -> Message:
    """Create an InternalMessage from raw message data."""
    # Handle content blocks
    processed_content: str | list[ContentBlock] | None = None

    if isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "text")
                if block_type == "text":
                    blocks.append(ContentBlock.from_text(block.get("text", "")))
                elif block_type == "tool_use":
                    blocks.append(ContentBlock.from_tool_use(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                    ))
                elif block_type == "tool_result":
                    blocks.append(ContentBlock.from_tool_result(
                        tool_use_id=block.get("tool_use_id", ""),
                        content=str(block.get("content", "")),
                    ))
                elif block_type == "thinking":
                    blocks.append(ContentBlock.from_thinking(block.get("thinking", "")))
        processed_content = blocks if blocks else None
    else:
        processed_content = content

    # Handle tool calls
    internal_tool_calls = []
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, dict):
                internal_tool_calls.append(ToolCall.from_dict(tc))

    return Message(
        role=role,
        content=processed_content,
        tool_calls=internal_tool_calls,
        tool_call_id=tool_call_id,
        name=name,
    )
