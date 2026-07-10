"""Thin NVIDIA chat-completions client. Forward requests, pass through responses."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class NvidiaResponse:
    status_code: int
    json_data: dict[str, Any] | None = None
    text: str = ""
    headers: httpx.Headers | None = None


class NvidiaClient:
    """Wraps NVIDIA's OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, timeout: float) -> None:
        self.chat_url = self._chat_url(base_url)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    @staticmethod
    def is_valid_key(api_key: str | None) -> bool:
        # Accept NVIDIA's real nvapi- keys, the nvda- shape from the spec,
        # and bare tokens — permissive, the upstream decides what's real.
        return bool(api_key)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, api_key: str, payload: dict[str, Any]) -> NvidiaResponse:
        response = await self._client.post(
            self.chat_url,
            headers=self._headers(api_key),
            json=payload,
        )
        return self._response_from_httpx(response)

    async def stream_chat(
        self, api_key: str, payload: dict[str, Any]
    ) -> tuple[int, httpx.Headers, AsyncIterator[bytes]]:
        request = self._client.build_request(
            "POST", self.chat_url, headers=self._headers(api_key), json=payload
        )
        response = await self._client.send(request, stream=True)

        async def iterator() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
            finally:
                await response.aclose()

        return response.status_code, response.headers, iterator()

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _chat_url(base_url: str) -> str:
        # Accept either the bare API root or a full /chat/completions URL.
        url = base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        return f"{url}/chat/completions"

    @staticmethod
    def _response_from_httpx(response: httpx.Response) -> NvidiaResponse:
        try:
            data = response.json()
        except ValueError:
            data = None
        return NvidiaResponse(
            status_code=response.status_code,
            json_data=data,
            text=response.text,
            headers=response.headers,
        )


def retry_after_seconds(headers: httpx.Headers | None, default: int) -> int:
    # Parse Retry-After defensively; upstream headers are not always polite.
    if headers is None:
        return default
    raw = headers.get("retry-after")
    if not raw:
        return default
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return default
