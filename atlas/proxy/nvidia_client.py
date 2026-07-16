from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Callable

import httpx


logger = logging.getLogger("atlas-proxy")

# Thin wrapper around NVIDIA's chat-completions endpoint.
@dataclass
class NvidiaResponse:
    status_code: int
    json_data: dict[str, Any] | None = None
    text: str = ""
    headers: httpx.Headers | None = None


class NvidiaClient:
    """NVIDIA chat-completions client with split timeout strategies.

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
        # concurrent requests over one connection and compresses headers; NVIDIA
        # supports it, so we negotiate it.
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )
        # Non-stream: flat total timeout, fast-fail, free the key.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            limits=limits,
            http2=True,
        )
        # Stream: generous read as a dead-stream backstop, no total cap.
        # The keepalive wrapper in the proxy keeps the downstream client alive
        # during reasoning-model thinking gaps; the read timeout only fires
        # when the upstream is genuinely silent past this window.
        self._stream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=connect_timeout, read=read_timeout),
            limits=limits,
            http2=True,
        )

    @staticmethod
    def is_valid_key(api_key: str | None) -> bool:
        # The proxy only treats NVIDIA as usable when the key looks like a real
        # nvapi-* credential.
        return bool(api_key and api_key.startswith("nvapi-"))

    async def close(self) -> None:
        await asyncio.gather(self._client.aclose(), self._stream_client.aclose())

    async def prewarm(self) -> None:
        """Warm the TLS/HTTP2 connection pool so the first real request skips
        the handshake. Best-effort: a failure here (NVIDIA unreachable, 405,
        auth rejection) is expected and harmless — we just want the TCP+TLS
        session established, not a successful chat. Runs both clients through
        a cheap GET in parallel.
        """
        async def _try(client: httpx.AsyncClient) -> None:
            try:
                # GET, not POST — we don't want to spend tokens or trigger a
                # real completion. The response status is irrelevant; the TLS
                # session landing in the pool is the point.
                await client.get(self.chat_url, headers={"User-Agent": "atlas-prewarm"})
            except Exception:
                pass

        await asyncio.gather(_try(self._client), _try(self._stream_client))

    async def chat(self, api_key: str, payload: dict[str, Any]) -> NvidiaResponse:
        response = await self._client.post(
            self.chat_url,
            headers=self._headers(api_key),
            json=payload,
        )
        return self._response_from_httpx(response)

    async def stream_chat(
        self,
        api_key: str,
        payload: dict[str, Any],
        rid: str = "",
        on_timeout: Callable[[], None] | None = None,
    ) -> tuple[int, httpx.Headers, AsyncIterator[bytes], str]:
        """Open a streaming chat request.

        Returns ``(status, headers, iterator, error_message)``. ``error_message``
        is the real upstream message (e.g. NVIDIA's "Validation: Temperature
        must be between 0 and 2, got 2.5") when ``status >= 400``, empty string
        otherwise — so the caller can surface it instead of the generic
        "upstream returned 400".

        ``on_timeout`` is invoked once if the upstream goes silent past the
        read deadline mid-stream — the proxy uses it to cool the key and
        record a failure, which the iterator's own except cannot do directly
        without a handle to the key store.
        """
        request = self._stream_client.build_request(
            "POST",
            self.chat_url,
            headers=self._headers(api_key),
            json=payload,
        )
        response = await self._stream_client.send(request, stream=True)

        # On a non-2xx upstream status the SSE body is usually a short JSON
        # error (e.g. NVIDIA's 400 "Validation: Temperature must be between
        # 0 and 2"). Drain it so the caller can surface the real message
        # instead of the generic "upstream returned 400". The stream is small
        # and terminal, so buffering it whole is safe.
        if response.status_code >= 400:
            error_body = b""
            try:
                async for chunk in response.aiter_bytes():
                    error_body += chunk
                    if len(error_body) > 4096:  # cap; error bodies are tiny
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
            try:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
            except httpx.TimeoutException as exc:
                # Mid-stream timeout (idle gap exceeded). Emit a terminal
                # OpenAI-shaped SSE error + [DONE] so the downstream adapter
                # and the client both see a clean end-of-stream instead of a
                # truncated connection with no final event. Distinguish the
                # timeout flavour so the message is actually useful for triage,
                # and thread rid so the client can correlate to a log line.
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


def _timeout_kind(exc: httpx.TimeoutException) -> str:
    """Map an httpx timeout subclass to a short diagnostic label."""
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
    """Pull a human message out of an upstream error body.

    NVIDIA's chat-completions errors come in a few shapes:
    - {"message": "Validation: Temperature must be between 0 and 2, got 2.5",
       "type": "Bad Request", "code": 400}
    - {"error": {"message": "...", "type": "...", "code": ...}}   (OpenAI shape)
    - {"detail": "..."}                                            (FastAPI shape)
    Fall back to the raw text if none match.
    """
    if not error_text:
        return "upstream error"
    try:
        data = json.loads(error_text)
    except (ValueError, TypeError):
        return error_text.strip()[:500] or "upstream error"
    if isinstance(data, dict):
        # NVIDIA native: top-level "message".
        msg = data.get("message")
        if isinstance(msg, str) and msg:
            return msg
        # OpenAI shape.
        err = data.get("error")
        if isinstance(err, dict):
            m = err.get("message")
            if isinstance(m, str) and m:
                return m
        # FastAPI shape.
        detail = data.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        # NVIDIA 429 shape: {"status":429,"title":"Too Many Requests"}.
        title = data.get("title")
        if isinstance(title, str) and title:
            return title
    return error_text.strip()[:500] or "upstream error"


async def _error_iterator(error_text: str) -> AsyncIterator[bytes]:
    """Replay a captured upstream error body as an OpenAI-shaped SSE error chunk.

    The streaming handlers consume the iterator's bytes through the SSE adapters,
    which already understand an ``{"error": {...}}`` chunk (see
    openai_sse_to_anthropic_sse / stream_router_sse). Emitting the real
    upstream message here means a 400 surfaces as
    "upstream returned 400: <Validation: ...>" instead of a bare
    "upstream returned 400".
    """
    message = _extract_error_message(error_text)
    err = {"error": {"message": message, "type": "upstream_error"}}
    yield f"data: {json.dumps(err)}\n\n".encode()
    yield b"data: [DONE]\n\n"
