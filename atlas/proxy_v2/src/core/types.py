"""
Canonical internal message types for Atlas Proxy v2.

This module defines the protocol-agnostic internal representation that all
protocol adapters convert to/from. This allows the proxy to:
1. Support multiple protocols (OpenAI, Anthropic) without code duplication
2. Preserve all content blocks (thinking, tool_use, etc.)
3. Enable provider abstraction
4. Maintain a clean separation of concerns

All adapters should convert to InternalRequest before processing,
and convert from InternalResponse for output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class Capability(Enum):
    """Model capabilities that providers can support."""
    CHAT = auto()           # Basic chat completion
    STREAMING = auto()      # Streaming responses
    TOOLS = auto()         # Tool/function calling
    THINKING = auto()       # Reasoning/thinking blocks
    VISION = auto()        # Image input
    EMBEDDINGS = auto()    # Text embeddings
    JSON_MODE = auto()      # Structured JSON output
    FUNCTION_CALLING = auto()  # Legacy function calling


class FinishReason(Enum):
    """Reasons why a response finished."""
    STOP = "stop"              # Normal completion
    LENGTH = "max_tokens"     # Hit max tokens
    TOOL_USE = "tool_use"    # Tool call requested
    CONTENT_FILTER = "content_filter"  # Content filtered
    ERROR = "error"           # Error occurred


class BlockType(Enum):
    """Types of content blocks."""
    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    SERVER_TOOL_USE = "server_tool_use"
    WEB_SEARCH_RESULT = "web_search_result"


@dataclass
class Logprobs:
    """Log probability information for token sampling."""
    tokens: list[str] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    top_logprobs: list[dict[str, float]] = field(default_factory=list)


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> Usage:
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class ImageSource:
    """Source of an image in a content block."""
    type: str = "base64"  # base64, url
    media_type: str = "image/png"
    data: str = ""  # base64 encoded data
    url: str = ""  # URL to image


@dataclass
class ContentBlock:
    """
    A single content block in a message.

    Supported types:
    - text: Plain text content
    - image: Image input (for multimodal)
    - tool_use: A tool call from the model
    - tool_result: A result from a tool execution
    - thinking: Reasoning/thinking content
    """
    type: BlockType = BlockType.TEXT
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict[str, Any]] = None
    tool_use_id: Optional[str] = None
    content: Optional[str] = None
    thinking: Optional[str] = None
    image: Optional[ImageSource] = None
    is_error: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def text(cls, content: str) -> ContentBlock:
        """Create a text content block."""
        return cls(type=BlockType.TEXT, text=content)

    @classmethod
    def thinking(cls, content: str) -> ContentBlock:
        """Create a thinking content block."""
        return cls(type=BlockType.THINKING, thinking=content)

    @classmethod
    def tool_use(cls, id: str, name: str, input: dict[str, Any]) -> ContentBlock:
        """Create a tool use content block."""
        return cls(
            type=BlockType.TOOL_USE,
            id=id,
            name=name,
            input=input,
        )

    @classmethod
    def tool_result(
        cls,
        tool_use_id: str,
        content: str,
        is_error: bool = False
    ) -> ContentBlock:
        """Create a tool result content block."""
        return cls(
            type=BlockType.TOOL_RESULT,
            tool_use_id=tool_use_id,
            content=content,
            is_error=is_error,
        )

    @classmethod
    def image(cls, source: ImageSource) -> ContentBlock:
        """Create an image content block."""
        return cls(type=BlockType.IMAGE, image=source)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {"type": self.type.value}

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
        if self.is_error:
            result["is_error"] = True
        if self.citations:
            result["citations"] = self.citations
        if self.image:
            result["source"] = {
                "type": self.image.type,
                "media_type": self.image.media_type,
                "data": self.image.data,
                "url": self.image.url,
            }

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentBlock:
        """Create from dictionary format."""
        block_type = BlockType(data.get("type", "text"))

        image_source = None
        if "source" in data:
            src = data["source"]
            image_source = ImageSource(
                type=src.get("type", "base64"),
                media_type=src.get("media_type", "image/png"),
                data=src.get("data", ""),
                url=src.get("url", ""),
            )

        return cls(
            type=block_type,
            text=data.get("text"),
            id=data.get("id"),
            name=data.get("name"),
            input=data.get("input"),
            tool_use_id=data.get("tool_use_id"),
            content=data.get("content"),
            thinking=data.get("thinking"),
            image=image_source,
            is_error=data.get("is_error", False),
            citations=data.get("citations", []),
        )


@dataclass
class ToolCall:
    """A tool call (function invocation) from the model."""
    id: str
    name: str
    arguments: dict[str, Any]
    index: int = 0

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
            arguments=args if isinstance(args, dict) else {},
            index=data.get("index", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
            "index": self.index,
        }

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Convert to Anthropic tool_use format."""
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.arguments,
        }


@dataclass
class ToolDefinition:
    """Definition of a tool available to the model."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    type: str = "function"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolDefinition:
        # Handle both OpenAI format (function) and Anthropic format (input_schema)
        func = data.get("function", {})
        return cls(
            name=func.get("name") or data.get("name", ""),
            description=func.get("description", ""),
            parameters=func.get("parameters", {}) or data.get("input_schema", {}),
            type=data.get("type", "function"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolChoice:
    """Controls how the model uses tools."""
    type: str = "auto"  # auto, any, tool, none
    name: Optional[str] = None
    disable_parallel_tool_use: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolChoice:
        if not isinstance(data, dict):
            return cls()

        return cls(
            type=data.get("type", "auto"),
            name=data.get("name"),
            disable_parallel_tool_use=data.get("disable_parallel_tool_use", False),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.name:
            result["name"] = self.name
        if self.disable_parallel_tool_use:
            result["disable_parallel_tool_use"] = True
        return result


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    content_blocks: list[ContentBlock] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    thinking: Optional[str] = None

    def get_text_content(self) -> str:
        """Extract plain text from content."""
        parts = []

        # Add thinking first if present
        if self.thinking:
            parts.append(self.thinking)

        # Add string content
        if self.content:
            parts.append(self.content)

        # Add content blocks
        for block in self.content_blocks:
            if block.type == BlockType.TEXT and block.text:
                parts.append(block.text)
            elif block.type == BlockType.THINKING and block.thinking:
                parts.append(block.thinking)

        return "\n".join(parts)

    def has_tool_calls(self) -> bool:
        """Check if message contains tool calls."""
        return bool(self.tool_calls or any(
            b.type == BlockType.TOOL_USE for b in self.content_blocks
        ))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            result["content"] = self.content

        if self.content_blocks:
            result["content"] = [b.to_dict() for b in self.content_blocks]

        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        if self.name:
            result["name"] = self.name

        if self.thinking:
            result["thinking"] = self.thinking

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        content = data.get("content", "")
        content_blocks: list[ContentBlock] = []
        tool_calls: list[ToolCall] = []

        # Parse content blocks
        if isinstance(content, list):
            for block_data in content:
                if isinstance(block_data, dict):
                    content_blocks.append(ContentBlock.from_dict(block_data))
            # Extract text from blocks for backward compatibility
            content = content_blocks[0].text if content_blocks else ""

        # Parse tool calls
        for tc_data in data.get("tool_calls", []):
            if isinstance(tc_data, dict):
                tool_calls.append(ToolCall.from_dict(tc_data))

        return cls(
            role=data.get("role", "user"),
            content=content,
            content_blocks=content_blocks,
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            thinking=data.get("thinking"),
        )


@dataclass
class RequestOptions:
    """Request-level options for LLM inference."""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[list[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    response_format: Optional[dict[str, Any]] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[ToolChoice] = None
    thinking: Optional[dict[str, Any]] = None  # For reasoning models
    logprobs: bool = False
    top_logprobs: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.top_k is not None:
            result["top_k"] = self.top_k
        if self.stop:
            result["stop"] = self.stop
        if self.frequency_penalty != 0.0:
            result["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            result["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            result["seed"] = self.seed
        if self.response_format:
            result["response_format"] = self.response_format
        if self.tools:
            result["tools"] = [t.to_dict() for t in self.tools]
        if self.tool_choice:
            result["tool_choice"] = self.tool_choice.to_dict()
        if self.thinking:
            result["thinking"] = self.thinking
        if self.logprobs:
            result["logprobs"] = True
            if self.top_logprobs > 0:
                result["top_logprobs"] = self.top_logprobs

        return result


@dataclass
class Request:
    """Canonical request representation."""
    model: str
    messages: list[Message]
    system: Optional[str] = None
    options: RequestOptions = field(default_factory=RequestOptions)
    stream: bool = False
    temperature: Optional[float] = None  # Legacy, use options
    max_tokens: Optional[int] = None      # Legacy, use options

    def get_effective_temperature(self) -> float:
        """Get temperature from options or legacy field."""
        if self.temperature is not None:
            return self.temperature
        return self.options.temperature

    def get_effective_max_tokens(self) -> int:
        """Get max_tokens from options or legacy field."""
        if self.max_tokens is not None:
            return self.max_tokens
        return self.options.max_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
            "stream": self.stream,
        }

        if self.system:
            result["system"] = self.system

        # Merge options
        result.update(self.options.to_dict())

        return result


@dataclass
class Response:
    """Canonical response representation."""
    id: str
    model: str
    content: str = ""
    content_blocks: list[ContentBlock] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = field(default_factory=Usage)
    logprobs: Optional[Logprobs] = None
    thinking: Optional[str] = None

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls or any(
            b.type == BlockType.TOOL_USE for b in self.content_blocks
        ))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {
            "id": self.id,
            "model": self.model,
            "finish_reason": self.finish_reason.value,
            "usage": self.usage.to_dict(),
        }

        if self.content:
            result["content"] = self.content

        if self.content_blocks:
            result["content_blocks"] = [b.to_dict() for b in self.content_blocks]

        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        if self.thinking:
            result["thinking"] = self.thinking

        if self.logprobs:
            result["logprobs"] = {
                "tokens": self.logprobs.tokens,
                "token_logprobs": self.logprobs.token_logprobs,
                "top_logprobs": self.logprobs.top_logprobs,
            }

        return result


@dataclass
class Model:
    """Model information."""
    id: str
    name: str
    provider: str
    capabilities: list[Capability] = field(default_factory=list)
    context_length: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, capability: Capability) -> bool:
        """Check if model supports a capability."""
        return capability in self.capabilities


@dataclass
class Provider:
    """Provider configuration."""
    name: str
    api_key: str
    base_url: str
    timeout: float = 120.0
    max_retries: int = 3
    capabilities: list[Capability] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, capability: Capability) -> bool:
        """Check if provider supports a capability."""
        return capability in self.capabilities
