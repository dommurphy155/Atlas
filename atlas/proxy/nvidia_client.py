from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx


# Thin wrapper around NVIDIA's chat-completions endpoint.
@dataclass
class NvidiaResponse:
    status_code: int
    json_data: dict[str, Any] | None = None
    text: str = ""
    headers: httpx.Headers | None = None


class NvidiaClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
    ) -> None:
        self.chat_url = self._chat_url(base_url)
        # Split timeouts: a flat `timeout` applies to every read, so a stalled
        # upstream (200 headers then silence, or a long mid-stream gap) holds a
        # key for the full window. connect fails fast; read bounds the idle gap
        # between chunks so a dead stream frees the key in ~read_timeout, not
        # ~timeout. `timeout` still caps write/pool as a backstop.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout,
                connect=connect_timeout,
                read=read_timeout,
            )
        )

    @staticmethod
    def is_valid_key(api_key: str | None) -> bool:
        # The proxy only treats NVIDIA as usable when the key looks like a real
        # nvapi-* credential.
        return bool(api_key and api_key.startswith("nvapi-"))

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, api_key: str, payload: dict[str, Any]) -> NvidiaResponse:
        response = await self._client.post(
            self.chat_url,
            headers=self._headers(api_key),
            json=payload,
        )
        return self._response_from_httpx(response)

    async def stream_chat(self, api_key: str, payload: dict[str, Any]) -> tuple[int, httpx.Headers, AsyncIterator[bytes]]:
        request = self._client.build_request(
            "POST",
            self.chat_url,
            headers=self._headers(api_key),
            json=payload,
        )
        response = await self._client.send(request, stream=True)

        async def iterator() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
            except httpx.TimeoutException:
                # Mid-stream read timeout (idle gap exceeded). Emit a terminal
                # OpenAI-shaped SSE error + [DONE] so the downstream adapter
                # and the client both see a clean end-of-stream instead of a
                # truncated connection with no final event.
                err = {
                    "error": {
                        "message": "upstream stream timed out (idle read)",
                        "type": "upstream_timeout",
                        "code": 504,
                    }
                }
                yield f"data: {json.dumps(err)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                await response.aclose()

        return response.status_code, response.headers, iterator()

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _chat_url(base_url: str) -> str:
        # Accept either the bare API root or a full /chat/completions URL so the
        # installer and runtime config can use whichever shape is easiest.
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
