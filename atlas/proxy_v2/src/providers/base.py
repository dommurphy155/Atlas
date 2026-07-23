"""
Provider abstraction layer for Atlas Proxy v2.

This module provides a clean abstraction for different LLM providers (NVIDIA, Anthropic,
OpenAI, etc.). Each provider implements a common interface, enabling:
- Easy addition of new providers
- Provider fallback/retry logic
- Unified request/response handling
- Capability-based provider selection
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator, Optional

from src.core.types import (
    Capability,
    Request,
    Response,
    Usage,
)


class ProviderCapability(Enum):
    """Provider-specific capabilities."""
    CHAT = auto()
    STREAMING = auto()
    TOOLS = auto()
    THINKING = auto()
    VISION = auto()
    EMBEDDINGS = auto()
    JSON_MODE = auto()
    FUNCTION_CALLING = auto()


@dataclass
class ProviderConfig:
    """Configuration for a provider instance."""
    name: str
    api_key: str
    base_url: str
    timeout: float = 120.0
    max_retries: int = 3
    capabilities: list[ProviderCapability] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Provider-specific options
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from a chat completion."""
    response: Response
    provider_name: str
    model: str
    raw_response: dict[str, Any]


@dataclass
class StreamResponse:
    """Iterator for streaming responses."""
    iterator: AsyncIterator[Response]
    provider_name: str
    model: str


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, provider: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code


class ProviderTimeoutError(ProviderError):
    """Timeout error from provider."""

    def __init__(self, message: str, provider: str):
        super().__init__(message, provider, 504)


class ProviderAuthError(ProviderError):
    """Authentication error from provider."""

    def __init__(self, message: str, provider: str):
        super().__init__(message, provider, 401)


class ProviderRateLimitError(ProviderError):
    """Rate limit error from provider."""

    def __init__(self, message: str, provider: str, retry_after: Optional[int] = None):
        super().__init__(message, provider, 429)
        self.retry_after = retry_after


class Provider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement these methods to be used by the proxy.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    async def chat(self, request: Request) -> ChatResponse:
        """
        Execute a non-streaming chat completion.

        Args:
            request: The internal request

        Returns:
            ChatResponse with the model's response

        Raises:
            ProviderError: On provider errors
            ProviderTimeoutError: On timeout
            ProviderAuthError: On authentication failure
            ProviderRateLimitError: On rate limit
        """
        pass

    @abstractmethod
    async def stream_chat(self, request: Request) -> StreamResponse:
        """
        Execute a streaming chat completion.

        Args:
            request: The internal request (with stream=True)

        Returns:
            StreamResponse with streaming iterator

        Raises:
            ProviderError: On provider errors
            ProviderTimeoutError: On timeout
            ProviderAuthError: On authentication failure
            ProviderRateLimitError: On rate limit
        """
        pass

    @abstractmethod
    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if provider supports a capability."""
        pass

    @abstractmethod
    def get_models(self) -> list[str]:
        """Get list of available models for this provider."""
        pass

    async def close(self) -> None:
        """Clean up provider resources."""
        pass

    def __repr__(self) -> str:
        return f"<Provider: {self.name}>"


class MultiProvider(Provider):
    """
    Wrapper for using multiple providers with fallback.

    Tries providers in order until one succeeds.
    """

    def __init__(self, providers: list[Provider], config: Optional[ProviderConfig] = None):
        # Create a combined config
        if config is None:
            config = ProviderConfig(
                name="multi",
                api_key="",
                base_url="",
                capabilities=[],  # Combined capabilities
            )

        super().__init__(config)
        self.providers = providers

    async def chat(self, request: Request) -> ChatResponse:
        """Try each provider until one succeeds."""
        last_error: Optional[Exception] = None

        for provider in self.providers:
            try:
                return await provider.chat(request)
            except ProviderRateLimitError as e:
                # Don't retry rate limits immediately
                last_error = e
                continue
            except ProviderError as e:
                last_error = e
                continue

        raise last_error or ProviderError("All providers failed", "multi")

    async def stream_chat(self, request: Request) -> StreamResponse:
        """Try each provider until one succeeds."""
        last_error: Optional[Exception] = None

        for provider in self.providers:
            try:
                return await provider.stream_chat(request)
            except ProviderRateLimitError as e:
                last_error = e
                continue
            except ProviderError as e:
                last_error = e
                continue

        raise last_error or ProviderError("All providers failed", "multi")

    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if any provider supports the capability."""
        return any(p.supports_capability(capability) for p in self.providers)

    def get_models(self) -> list[str]:
        """Get combined list of models."""
        models = []
        seen = set()
        for provider in self.providers:
            for model in provider.get_models():
                if model not in seen:
                    seen.add(model)
                    models.append(model)
        return models


# Provider class registry
_PROVIDER_CLASSES: dict[str, type[Provider]] = {}


def register_provider(name: str):
    """Decorator to register a provider class."""
    def decorator(cls: type[Provider]):
        _PROVIDER_CLASSES[name] = cls
        return cls
    return decorator


def create_provider(name: str, config: ProviderConfig) -> Provider:
    """Create a provider instance by name."""
    if name not in _PROVIDER_CLASSES:
        raise ValueError(f"Unknown provider: {name}. Available: {list(_PROVIDER_CLASSES.keys())}")
    return _PROVIDER_CLASSES[name](config)
