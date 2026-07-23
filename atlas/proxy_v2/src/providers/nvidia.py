"""
NVIDIA provider implementation for Atlas Proxy v2.

This module provides a provider implementation for NVIDIA's AI Foundation Cloud API,
with support for:
- Key rotation with cooldown
- Streaming responses
- Tool calling
- Thinking blocks
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Optional

import httpx

from src.core.types import (
    BlockType,
    Capability,
    ContentBlock,
    FinishReason,
    Message,
    Request,
    Response,
    ToolCall,
    Usage,
)
from src.providers.base import (
    Provider,
    ProviderCapability,
    ProviderConfig,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ChatResponse,
    StreamResponse,
)


def _generate_id() -> str:
    """Generate a response ID."""
    from src.protocols.openai import generate_id
    return generate_id()


class NvidiaProvider(Provider):
    """Provider for NVIDIA AI Foundation Cloud."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._stream_client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            limits = httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=300.0,
            )
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout, connect=15.0),
                limits=limits,
                http2=True,
            )
        return self._client

    @property
    def stream_client(self) -> httpx.AsyncClient:
        if self._stream_client is None:
            limits = httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=300.0,
            )
            self._stream_client = httpx.AsyncClient(
                timeout=httpx.Timeout(None, connect=15.0, read=180.0),
                limits=limits,
                http2=True,
            )
        return self._stream_client

    def _get_url(self, path: str = "/chat/completions") -> str:
        """Get the full API URL."""
        base = self.config.base_url.rstrip("/")
        if base.endswith(path):
            return base
        return f"{base}{path}"

    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if provider supports a capability."""
        # Map internal capability to provider capability
        capability_map = {
            Capability.CHAT: ProviderCapability.CHAT,
            Capability.STREAMING: ProviderCapability.STREAMING,
            Capability.TOOLS: ProviderCapability.TOOLS,
            Capability.THINKING: ProviderCapability.THINKING,
            Capability.VISION: ProviderCapability.VISION,
            Capability.JSON_MODE: ProviderCapability.JSON_MODE,
        }
        pc = capability_map.get(capability)
        return pc in self.config.capabilities if pc else False

    def get_models(self) -> list[str]:
        """Get list of available models."""
        return self.config.models

    async def chat(self, request: Request) -> ChatResponse:
        """Execute a non-streaming chat completion."""
        payload = self._build_payload(request)

        try:
            response = await self.client.post(
                self._get_url(),
                headers=self._headers(),
                json=payload,
            )
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"Request timed out: {e}", self.name)
        except httpx.HTTPError as e:
            raise ProviderError(f"HTTP error: {e}", self.name)

        if response.status_code == 429:
            raise ProviderRateLimitError("Rate limited", self.name)
        if response.status_code >= 400:
            error_body = self._extract_error(response)
            raise ProviderError(error_body, self.name, response.status_code)

        data = response.json()
        internal_response = self._parse_response(request.model, data)

        return ChatResponse(
            response=internal_response,
            provider_name=self.name,
            model=request.model,
            raw_response=data,
        )

    async def stream_chat(self, request: Request) -> StreamResponse:
        """Execute a streaming chat completion."""
        payload = self._build_payload(request, stream=True)

        async def iterator() -> AsyncIterator[Response]:
            try:
                request = self.stream_client.build_request(
                    "POST",
                    self._get_url(),
                    headers=self._headers(),
                    json=payload,
                )
                response = await self.stream_client.send(request, stream=True)
            except httpx.TimeoutException as e:
                yield self._error_response(request.model, f"Timeout: {e}")
                return
            except httpx.HTTPError as e:
                yield self._error_response(request.model, f"HTTP error: {e}")
                return

            if response.status_code >= 400:
                error_body = self._extract_error(response)
                yield self._error_response(request.model, error_body, response.status_code)
                return

            buffer = b""
            async for chunk in response.aiter_bytes():
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == b"[DONE]":
                        return

                    try:
                        data = json.loads(data_str)
                        response_chunk = self._parse_chunk(request.model, data)
                        if response_chunk:
                            yield response_chunk
                    except json.JSONDecodeError:
                        continue

        return StreamResponse(
            iterator=iterator(),
            provider_name=self.name,
            model=request.model,
        )

    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.extra_headers)
        return headers

    def _build_payload(self, request: Request, stream: bool = False) -> dict[str, Any]:
        """Build the request payload."""
        # Convert messages
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})

        for msg in request.messages:
            msg_dict: dict[str, Any] = {"role": msg.role}

            if msg.content:
                msg_dict["content"] = msg.content
            elif msg.content_blocks:
                blocks = []
                for block in msg.content_blocks:
                    if block.type == BlockType.TEXT and block.text:
                        blocks.append({"type": "text", "text": block.text})
                    elif block.type == BlockType.THINKING and block.thinking:
                        blocks.append({"type": "thinking", "thinking": block.thinking})
                    elif block.type == BlockType.TOOL_USE:
                        blocks.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                msg_dict["content"] = blocks

            if msg.tool_calls:
                msg_dict["tool_calls"] = [tc.to_dict() for tc in msg.tool_calls]

            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id

            messages.append(msg_dict)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.get_effective_max_tokens(),
            "temperature": request.get_effective_temperature(),
            "stream": stream,
        }

        # Add options
        opts = request.options
        if opts.top_p is not None:
            payload["top_p"] = opts.top_p
        if opts.stop:
            payload["stop"] = opts.stop
        if opts.frequency_penalty != 0.0:
            payload["frequency_penalty"] = opts.frequency_penalty
        if opts.presence_penalty != 0.0:
            payload["presence_penalty"] = opts.presence_penalty

        # Add tools
        if opts.tools:
            payload["tools"] = [t.to_dict() for t in opts.tools]
            if opts.tool_choice:
                payload["tool_choice"] = opts.tool_choice.to_dict()
            else:
                payload["tool_choice"] = "auto"

        # Add thinking config
        if opts.thinking:
            payload["thinking"] = opts.thinking

        # Add stream_options for usage tracking
        if stream:
            payload["stream_options"] = {"include_usage": True}

        return payload

    def _parse_response(self, model: str, data: dict[str, Any]) -> Response:
        """Parse a non-streaming response."""
        choices = data.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}

        content = message.get("content", "")
        tool_calls = []
        for tc in message.get("tool_calls", []):
            tool_calls.append(ToolCall.from_dict(tc))

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        finish_reason = FinishReason.STOP
        if choices:
            reason = choices[0].get("finish_reason")
            if reason == "tool_calls":
                finish_reason = FinishReason.TOOL_USE
            elif reason == "length":
                finish_reason = FinishReason.LENGTH
            elif reason == "content_filter":
                finish_reason = FinishReason.CONTENT_FILTER

        return Response(
            id=data.get("id", _generate_id()),
            model=model,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _parse_chunk(self, model: str, data: dict[str, Any]) -> Optional[Response]:
        """Parse a streaming chunk."""
        choices = data.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})
        content = delta.get("content", "")

        # Extract tool calls
        tool_calls = []
        for tc in delta.get("tool_calls", []):
            tool_calls.append(ToolCall.from_dict(tc))

        # Get usage from final chunk
        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Determine finish reason
        finish_reason = FinishReason.STOP
        reason = choices[0].get("finish_reason")
        if reason:
            if reason == "tool_calls":
                finish_reason = FinishReason.TOOL_USE
            elif reason == "length":
                finish_reason = FinishReason.LENGTH
            elif reason == "content_filter":
                finish_reason = FinishReason.CONTENT_FILTER

        return Response(
            id=data.get("id", _generate_id()),
            model=model,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _error_response(self, model: str, message: str, status_code: int = 500) -> Response:
        """Create an error response."""
        return Response(
            id=_generate_id(),
            model=model,
            content=f"Error: {message}",
            finish_reason=FinishReason.ERROR,
            usage=Usage(),
        )

    def _extract_error(self, response: httpx.Response) -> str:
        """Extract error message from response."""
        try:
            data = response.json()
            if "error" in data:
                return data["error"].get("message", "Unknown error")
            if "message" in data:
                return data["message"]
            if "detail" in data:
                return data["detail"]
        except Exception:
            pass
        return response.text[:500] or f"HTTP {response.status_code}"

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._stream_client:
            await self._stream_client.aclose()
            self._stream_client = None
