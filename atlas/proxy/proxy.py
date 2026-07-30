"""ProxyCore — HTTP forwarding, SSE streaming, connection pool."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Dict, Optional

import httpx
from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import (
    CONNECT_TIMEOUT,
    HEALTH_CHECK_INTERVAL,
    KEEPALIVE_EXPIRY,
    MAX_CONNECTIONS,
    MAX_KEEPALIVE_CONNECTIONS,
    MAX_RETRIES,
    POOL_TIMEOUT,
    PREWARM_INTERVAL,
    READ_TIMEOUT,
    RETRY_STATUSES,
    WRITE_TIMEOUT,
    get_models_endpoint,
    get_logger,
)
from .keypool import KeyPool
from .translation import translate_stream_openai_to_anthropic, translate_stream_anthropic_to_openai
from .utils import dumps, is_openai_done_frame

log = get_logger(__name__)


class ProxyCore:
    def __init__(self, pool: KeyPool) -> None:
        self.pool = pool
        self.client: Optional[httpx.AsyncClient] = None
        self._prewarm_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        limits = httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=KEEPALIVE_EXPIRY,
        )
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=READ_TIMEOUT,
            write=WRITE_TIMEOUT,
            pool=POOL_TIMEOUT,
        )
        self.client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=True,
            headers={
                "HTTP-Referer": "https://localhost:8788",
                "X-Title": "OpenRouter-Translation-Proxy",
            },
            follow_redirects=True,
        )
        await self._prewarm()
        self._prewarm_task = asyncio.create_task(self._prewarm_loop())
        self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        for task in (self._prewarm_task, self._health_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self.client:
            await self.client.aclose()
            self.client = None

    async def _prewarm(self) -> None:
        if not self.client or not self.pool.total:
            return
        key, _ = self.pool.next_key()
        try:
            resp = await self.client.get(
                get_models_endpoint(),
                headers={"Authorization": f"Bearer {key}"},
                timeout=10.0,
            )
            await resp.aread()
            log.info("TLS pre-warm complete (status=%s)", resp.status_code)
        except Exception as e:
            log.warning("Pre-warm failed (non-fatal): %s", e)

    async def _prewarm_loop(self) -> None:
        while True:
            await asyncio.sleep(PREWARM_INTERVAL)
            await self._prewarm()

    async def _health_loop(self) -> None:
        """Periodically log key-pool health; recovery is lazy inside next_key."""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            s = self.pool.stats()
            log.info(
                "Key health: total=%d healthy=%d cooling=%d suspended=%d",
                s["total"],
                s["healthy"],
                s["cooling"],
                s["suspended"],
            )

    def _headers(
        self, key: str, extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:8788",
            "X-Title": "OpenRouter-Translation-Proxy",
        }
        if extra:
            h.update(extra)
        return h

    async def forward(
        self,
        method: str,
        url: str,
        body: Optional[bytes] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
        request_id: str = "",
    ) -> Response | StreamingResponse:
        assert self.client is not None
        last_error: Optional[Exception] = None
        last_status = 502

        for attempt in range(MAX_RETRIES + 1):
            key, key_idx = self.pool.next_key()
            headers = self._headers(key, extra_headers)
            t0 = time.perf_counter()
            try:
                if stream:
                    result = await self._stream_forward(
                        method, url, headers, body, key_idx, request_id, t0
                    )
                    if isinstance(result, StreamingResponse):
                        return result
                    if (
                        result.status_code in RETRY_STATUSES
                        and attempt < MAX_RETRIES
                    ):
                        log.warning(
                            "req=%s key_idx=%d stream-status=%d attempt=%d — retrying",
                            request_id,
                            key_idx,
                            result.status_code,
                            attempt + 1,
                        )
                        last_status = result.status_code
                        continue
                    result.headers["x-request-id"] = request_id
                    return result

                resp = await self.client.request(
                    method, url, headers=headers, content=body
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                status = resp.status_code

                if status in RETRY_STATUSES and attempt < MAX_RETRIES:
                    await self.pool.mark_error(key_idx, status)
                    log.warning(
                        "req=%s key_idx=%d status=%d attempt=%d — retrying",
                        request_id,
                        key_idx,
                        status,
                        attempt + 1,
                    )
                    last_status = status
                    continue

                if status >= 400:
                    await self.pool.mark_error(key_idx, status)
                else:
                    await self.pool.mark_success(key_idx, latency_ms)

                out_headers = {
                    k: v
                    for k, v in resp.headers.items()
                    if k.lower()
                    not in (
                        "transfer-encoding",
                        "content-encoding",
                        "content-length",
                        "connection",
                    )
                }
                out_headers["x-request-id"] = request_id
                data = resp.content
                log.info(
                    "req=%s key_idx=%d status=%d latency=%.0fms bytes=%d",
                    request_id,
                    key_idx,
                    status,
                    latency_ms,
                    len(data),
                )
                return Response(
                    content=data,
                    status_code=status,
                    headers=out_headers,
                    media_type=resp.headers.get("content-type"),
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                await self.pool.mark_error(key_idx, 599)
                last_error = e
                last_status = (
                    504 if isinstance(e, httpx.TimeoutException) else 502
                )
                log.warning(
                    "req=%s key_idx=%d transport error attempt=%d: %s",
                    request_id,
                    key_idx,
                    attempt + 1,
                    e,
                )
                if attempt >= MAX_RETRIES:
                    break
                continue

        msg = f"proxy upstream error after retries: {last_error}"
        log.error("req=%s %s", request_id, msg)
        return JSONResponse(
            status_code=last_status,
            content={
                "error": {
                    "message": msg,
                    "type": "proxy_error",
                    "code": last_status,
                }
            },
            headers={"x-request-id": request_id},
        )

    async def _stream_forward(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes],
        key_idx: int,
        request_id: str,
        t0: float,
    ) -> StreamingResponse | Response:
        """
        Open upstream SSE. Retry is handled by forward() only when we return
        a plain Response (failure before any client bytes). Once we return
        StreamingResponse, chunks are flushed immediately and sanitized.
        """
        assert self.client is not None
        req = self.client.build_request(
            method, url, headers=headers, content=body
        )
        try:
            upstream = await self.client.send(req, stream=True)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            await self.pool.mark_error(key_idx, 599)
            log.warning(
                "req=%s key_idx=%d stream connect error: %s",
                request_id,
                key_idx,
                e,
            )
            return Response(
                content=dumps(
                    {
                        "error": {
                            "message": f"upstream connect error: {e}",
                            "type": "proxy_error",
                        }
                    }
                ),
                status_code=502,
                media_type="application/json",
            )

        status = upstream.status_code

        if status in RETRY_STATUSES or status >= 400:
            await self.pool.mark_error(key_idx, status)
            data = await upstream.aread()
            await upstream.aclose()
            return Response(
                content=data,
                status_code=status,
                media_type=upstream.headers.get(
                    "content-type", "application/json"
                ),
            )

        await self.pool.mark_success(key_idx, (time.perf_counter() - t0) * 1000)
        log.info(
            "req=%s key_idx=%d status=%d stream=1", request_id, key_idx, status
        )

        out_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower()
            not in (
                "transfer-encoding",
                "content-encoding",
                "content-length",
                "connection",
            )
        }
        out_headers["content-type"] = "text/event-stream; charset=utf-8"
        out_headers["cache-control"] = "no-cache, no-transform"
        out_headers["x-accel-buffering"] = "no"
        out_headers["connection"] = "keep-alive"
        out_headers["x-request-id"] = request_id

        # Determine if we need to translate the stream
        # Check if request is for chat/completions (OpenAI) → translate to Anthropic
        # or messages (Anthropic) → translate to OpenAI
        translate_openai_to_anthropic = "/chat/completions" in url
        translate_anthropic_to_openai = "/messages" in url

        async def event_generator() -> AsyncIterator[bytes]:
            """
            SSE with optional translation:
              • drop OpenAI-style `data: [DONE]` trailers
              • drop bare `event: data` frames
              • translate between OpenAI and Anthropic SSE formats when needed
              • yield every chunk immediately (no re-buffering)
            """
            buf = b""
            try:
                if translate_openai_to_anthropic:
                    # Use translation generator
                    async for chunk in translate_stream_openai_to_anthropic(upstream):
                        yield chunk
                    return

                if translate_anthropic_to_openai:
                    # Use translation generator
                    async for chunk in translate_stream_anthropic_to_openai(upstream):
                        yield chunk
                    return

                # Pass-through (no translation)
                async for raw in upstream.aiter_raw():
                    if not raw:
                        continue
                    buf += raw
                    while True:
                        sep = buf.find(b"\n\n")
                        if sep < 0:
                            sep = buf.find(b"\r\n\r\n")
                            if sep < 0:
                                break
                            frame = buf[:sep]
                            buf = buf[sep + 4 :]
                        else:
                            frame = buf[:sep]
                            buf = buf[sep + 2 :]

                        if is_openai_done_frame(frame):
                            continue
                        yield frame + b"\n\n"
                if buf.strip() and not is_openai_done_frame(buf):
                    yield buf if buf.endswith(b"\n\n") else buf + b"\n\n"
            except (httpx.ReadError, httpx.StreamError, asyncio.CancelledError):
                pass
            except Exception as e:
                log.warning(
                    "req=%s stream mid-body error: %s", request_id, e
                )
            finally:
                try:
                    await upstream.aclose()
                except Exception:
                    pass

        return StreamingResponse(
            event_generator(),
            status_code=status,
            headers=out_headers,
            media_type="text/event-stream",
        )
