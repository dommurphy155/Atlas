from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Callable

import httpx


logger = logging.getLogger("atlas-proxy")

# Thin wrapper around OpenRouter's chat-completions endpoint.
@dataclass
class OpenRouterResponse:
    status_code: int
    json_data: dict[str, Any] | None = None
    text: str = ""
    headers: httpx.Headers | None = None


class OpenRouterClient:
    """OpenRouter chat-completions client with split timeout strategies.

    Non-streaming requests use a flat total ``timeout`` so a hung response
    fails fast and frees the key. Streaming requests need a different model:
    reasoning models sit silent for long stretches (prefill, thinking), so a
    per-read deadline that's too short kills healthy streams, while no cap at
    all lets a dead upstream hold a key forever. The stream client therefore
    uses a short ``connect`` and a generous ``read`` (the dead-stream
    backstop) with no total cap — the proxy bounds the *stream* lifetime via
    its keepalive wrapper, not httpx.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
    ) -> None:
        self.chat_url = self._chat_url(base_url)
        # Shared connection-pool limits. A warm pool with long keepalive expiry
        # means repeated requests reuse the TLS session instead of paying the
        # handshake every time — the main "feels slow" fix. HTTP/2 multiplexes
        # concurrent requests over one connection and compresses headers; OpenRouter
        # supports it, so we negotiate it. 5-minute keepalive expiry so the pool
        # survives a coffee break — idle gaps between turns no longer cost a
        # cold TLS handshake (~0.4s) on the next request.
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=300.0,
        )
        # Non-stream: flat total timeout, fast-fail, free the key.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            limits=limits,
            http2=True,
        )
        # Stream: generous read as a dead-stream backstop, no total cap.
        self._stream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=connect_timeout, read=read_timeout),
            limits=limits,
            http2=True,
        )

    @staticmethod
    def is_valid_key(api_key: str | None) -> bool:
        # OpenRouter accepts any valid API key (non-empty, not just "sk-or-")
        return bool(api_key and len(api_key) > 10)

    async def close(self) -> None:
        await asyncio.gather(self._client.aclose(), self._stream_client.aclose())

    async def prewarm(self) -> None:
        """Warm the TLS/HTTP2 connection pool so the first real request skips
        the handshake. Best-effort: a failure here (OpenRouter unreachable, 405,
        auth rejection) is expected and harmless — we just want the TCP+TLS
        session established, not a successful chat. Runs both clients through
        a cheap GET in parallel.
        """
        async def _try(client: httpx.AsyncClient) -> None:
            try:
                await client.get(self.chat_url, headers={"User-Agent": "atlas-prewarm"})
            except Exception:
                pass

        await asyncio.gather(_try(self._client), _try(self._stream_client))

    async def chat(self, api_key: str, payload: dict[str, Any], timings: dict[str, float] | None = None) -> OpenRouterResponse:
        import time as _t
        _t_send = _t.monotonic()
        response = await self._client.post(
            self.chat_url,
            headers=self._headers(api_key),
            json=payload,
        )
        if timings is not None:
            timings["upstream"] = max(0.0, _t.monotonic() - _t_send)
            timings.setdefault("stream", 0.0)
        return self._response_from_httpx(response)

    async def stream_chat(
        self,
        api_key: str,
        payload: dict[str, Any],
        rid: str = "",
        on_timeout: Callable[[], None] | None = None,
        timings: dict[str, float] | None = None,
    ) -> tuple[int, httpx.Headers, AsyncIterator[bytes], str]:
        """Streaming chat. If ``timings`` is a dict, it is populated with
        ``upstream`` (send->first byte), ``ttft`` (request start->first byte)
        and ``stream`` (first byte->last byte) in seconds, for the proxy's log
        line. The handler seeds ``__started`` (request-received monotonic) into
        ``timings`` so ttft can span queue+preprocess+upstream. Monotonic clock."""
        import time as _t
        _t_send = _t.monotonic()
        if timings is not None:
            timings.setdefault("upstream", 0.0)
            timings.setdefault("stream", 0.0)
            timings.setdefault("ttft", 0.0)

        request = self._stream_client.build_request(
            "POST",
            self.chat_url,
            headers=self._headers(api_key),
            json=payload,
        )
        response = await self._stream_client.send(request, stream=True)

        if response.status_code >= 400:
            error_body = b""
            try:
                async for chunk in response.aiter_bytes():
                    error_body += chunk
                    if len(error_body) > 4096:
                        break
            finally:
                await response.aclose()
            error_text = ""
            try:
                error_text = error_body.decode("utf-8", errors="replace")
            except Exception:
                pass
            message = _extract_error_message(error_text)
            logger.warning("<%s upstream %d: %s", rid, response.status_code, message)
            return response.status_code, response.headers, _error_iterator(error_text), message

        async def iterator() -> AsyncIterator[bytes]:
            _first = None
            try:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        if _first is None:
                            _first = _t.monotonic()
                            if timings is not None:
                                timings["upstream"] = max(0.0, _first - _t_send)
                                _started = timings.get("__started")
                                if _started is not None:
                                    timings["ttft"] = max(0.0, _first - _started)
                        if timings is not None:
                            timings["stream"] = max(0.0, _t.monotonic() - _first)
                        yield chunk
            except httpx.TimeoutException as exc:
                kind = _timeout_kind(exc)
                if on_timeout is not None:
                    try:
                        on_timeout()
                    except Exception:
                        logger.warning("on_timeout callback failed for %s", rid)
                logger.warning("<%s upstream %s (mid-stream)", rid, kind)
                err = {
                    "error": {
                        "message": f"upstream stream timed out ({kind})",
                        "type": "upstream_timeout",
                        "code": 504,
                        "rid": rid,
                    }
                }
                yield f"data: {json.dumps(err)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                await response.aclose()

        return response.status_code, response.headers, iterator(), ""

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://atlas.local",
            "X-Title": "Atlas Proxy",
        }

    @staticmethod
    def _chat_url(base_url: str) -> str:
        url = base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        return f"{url}/chat/completions"

    @staticmethod
    def _response_from_httpx(response: httpx.Response) -> OpenRouterResponse:
        try:
            data = response.json()
        except ValueError:
            data = None
        return OpenRouterResponse(
            status_code=response.status_code,
            json_data=data,
            text=response.text,
            headers=response.headers,
        )


def _timeout_kind(exc: httpx.TimeoutException) -> str:
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "idle_read"
    if isinstance(exc, httpx.PoolTimeout):
        return "pool_timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "write_timeout"
    return "timeout"


def _extract_error_message(error_text: str) -> str:
    """Pull a human message out of an upstream error body."""
    if not error_text:
        return "upstream error"
    try:
        data = json.loads(error_text)
    except (ValueError, TypeError):
        return error_text.strip()[:500] or "upstream error"
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, str) and msg:
            return msg
        err = data.get("error")
        if isinstance(err, dict):
            m = err.get("message")
            if isinstance(m, str) and m:
                return m
        detail = data.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    return error_text.strip()[:500] or "upstream error"


async def _error_iterator(error_text: str) -> AsyncIterator[bytes]:
    message = _extract_error_message(error_text)
    err = {"error": {"message": message, "type": "upstream_error"}}
    yield f"data: {json.dumps(err)}\n\n".encode()
    yield b"data: [DONE]\n\n"
