"""Protocol adapter base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Optional

from src.core.types import Request, Response


class ProtocolType(Enum):
    """Supported protocol types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class ProtocolRequest:
    """Wrapper for protocol-specific request data."""
    protocol: ProtocolType
    raw_request: dict[str, Any]
    headers: dict[str, str]


@dataclass
class ProtocolResponse:
    """Wrapper for protocol-specific response data."""
    protocol: ProtocolType
    raw_response: dict[str, Any]
    status_code: int = 200


@dataclass
class StreamChunk:
    """A streaming response chunk."""
    data: dict[str, Any]
    event: Optional[str] = None  # For SSE events


class ProtocolAdapter(ABC):
    """
    Base class for protocol adapters.

    Each adapter is responsible for converting between external protocol
    formats and the internal canonical format.
    """

    def __init__(self, protocol: ProtocolType):
        self.protocol = protocol

    @abstractmethod
    def parse_request(self, data: dict[str, Any]) -> Request:
        """Parse external request to internal format."""
        pass

    @abstractmethod
    def format_response(self, response: Response) -> dict[str, Any]:
        """Format internal response to external format."""
        pass

    @abstractmethod
    def format_error(self, error: Exception) -> dict[str, Any]:
        """Format error to external format."""
        pass

    @abstractmethod
    def format_stream_chunk(self, chunk: Response) -> bytes:
        """Format streaming chunk to external format."""
        pass

    @abstractmethod
    def parse_stream_chunk(self, data: bytes) -> Optional[dict[str, Any]]:
        """Parse streaming chunk from external format."""
        pass


class OpenAIAdapter(ProtocolAdapter):
    """Adapter for OpenAI API."""

    def __init__(self):
        super().__init__(ProtocolType.OPENAI)

    def parse_request(self, data: dict[str, Any]) -> Request:
        from src.protocols.openai import parse_openai_request
        return parse_openai_request(data)

    def format_response(self, response: Response) -> dict[str, Any]:
        from src.protocols.openai import format_openai_response
        return format_openai_response(response)

    def format_error(self, error: Exception) -> dict[str, Any]:
        from src.protocols.openai import format_openai_error
        return format_openai_error(error)

    def format_stream_chunk(self, chunk: Response) -> bytes:
        from src.protocols.openai import format_openai_stream_chunk
        return format_openai_stream_chunk(chunk)

    def parse_stream_chunk(self, data: bytes) -> Optional[dict[str, Any]]:
        from src.protocols.openai import parse_openai_stream_chunk
        return parse_openai_stream_chunk(data)


class AnthropicAdapter(ProtocolAdapter):
    """Adapter for Anthropic API."""

    def __init__(self):
        super().__init__(ProtocolType.ANTHROPIC)

    def parse_request(self, data: dict[str, Any]) -> Request:
        from src.protocols.anthropic import parse_anthropic_request
        return parse_anthropic_request(data)

    def format_response(self, response: Response) -> dict[str, Any]:
        from src.protocols.anthropic import format_anthropic_response
        return format_anthropic_response(response)

    def format_error(self, error: Exception) -> dict[str, Any]:
        from src.protocols.anthropic import format_anthropic_error
        return format_anthropic_error(error)

    def format_stream_chunk(self, chunk: Response) -> bytes:
        from src.protocols.anthropic import format_anthropic_stream_event
        return format_anthropic_stream_event(chunk)

    def parse_stream_chunk(self, data: bytes) -> Optional[dict[str, Any]]:
        # Anthropic uses SSE events, handled differently
        return None


# Registry of adapters
ADAPTERS: dict[ProtocolType, type[ProtocolAdapter]] = {
    ProtocolType.OPENAI: OpenAIAdapter,
    ProtocolType.ANTHROPIC: AnthropicAdapter,
}


def get_adapter(protocol: ProtocolType) -> ProtocolAdapter:
    """Get adapter instance for protocol."""
    adapter_class = ADAPTERS.get(protocol)
    if not adapter_class:
        raise ValueError(f"Unknown protocol: {protocol}")
    return adapter_class()
