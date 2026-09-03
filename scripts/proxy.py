"""ProxyCore — HTTP forwarding, SSE streaming, connection pool."""

from __future__ import annotations

import asyncio
import json as _json
import time
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx
from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import (
    CONNECT_TIMEOUT,
    get_force_default_model,
    FREE_MODEL_MAX_CONCURRENT,
    HEALTH_CHECK_INTERVAL,
    KEEPALIVE_EXPIRY,
    MAX_CONNECTIONS,
    MAX_KEEPALIVE_CONNECTIONS,
    MAX_RETRIES,
    MAX_RESPONSE_BYTES,
    get_models_url,
    POOL_TIMEOUT,
    PREWARM_INTERVAL,
    PROXY_KEEPALIVE_SECONDS,
    READ_TIMEOUT,
    RETRY_STATUSES,
    UPSTREAM_REFERER,
    UPSTREAM_TITLE,
    WRITE_TIMEOUT,
    get_default_model,
    is_hf_rate_limit_error,
    is_hf_key_invalid,
    retire_and_remove_hf_key,
    get_logger,
)
from .keypool import KeyPool
from .utils import dumps, is_openai_done_frame, loads
from . import prettylog as pl

log = get_logger(__name__)


async def _quiet_cleanup(fn, *args, what: str = "cleanup", rid: str = "") -> None:
    """Run a cleanup callable, logging (not swallowing) any failure at debug level.

    Used for best-effort resource teardown (aclose/release/semaphore) where a
    failure must not mask the primary error but should still be visible when
    debugging. Replaces bare `except Exception: pass` blocks.
    """
    try:
        result = fn(*args)
        if asyncio.iscoroutine(result):
            await result
    except Exception as e:  # noqa: BLE001 - intentional best-effort cleanup
        log.debug("req=%s %s failed: %s", rid, what, e)


def _enforce_default_model(body: Optional[bytes]) -> Optional[bytes]:
    """Authoritative provider/model override.

    Every outgoing upstream payload passes through here. When
    get_force_default_model() is enabled (default), the `model` field — if present
    in the JSON body — is unconditionally replaced with the active provider's
    hardcoded default model. This guarantees that client-supplied model
    names (e.g. "claude-sonnet-5") can never reach the upstream provider
    regardless of endpoint, streaming mode, or provider (OpenRouter /
    Hugging Face). Returns the body unchanged if it is None or not JSON.
    """
    if not get_force_default_model() or body is None:
        return body
    try:
        parsed = loads(body)
    except Exception:
        return body
    if not isinstance(parsed, dict):
        return body
    if "model" not in parsed:
        return body
    parsed["model"] = get_default_model()
    return dumps(parsed)


# --- atlas-sse-error-classification-patch v1 ---
def _classify_sse_frame(frame: bytes) -> Optional[dict]:
    """
    Structurally classify a single SSE frame as either a provider error
    event, or ordinary content (return None).

    This function NEVER substring-matches the raw frame text. It only
    inspects:
      - the SSE `event:` field name (e.g. "error")
      - the JSON payload of the `data:` field, and only well-known error
        shape keys within it: top-level "type"=="error", a nested "error"
        object, or an OpenAI-style top-level "error" object.

    Ordinary content frames (content_block_delta, message text, tool-call
    argument deltas, etc.) do not have this shape and are always returned
    as None, regardless of what English words their content contains
    (e.g. "concurrently", "worker", "capacity", "rpm" in model output).

    Returns a dict {"kind": str, "raw_type": str|None, "message": str,
    "code": Any} for a genuine structured error frame, else None.
    """
    event_name = None
    data_raw = None
    for line in frame.split(b"\n"):
        line = line.strip(b"\r")
        if line.startswith(b"event:"):
            event_name = line[len(b"event:"):].strip().decode("utf-8", "ignore")
        elif line.startswith(b"data:"):
            piece = line[len(b"data:"):].strip()
            data_raw = piece if data_raw is None else data_raw + piece

    is_named_error_event = event_name == "error"

    obj = None
    if data_raw and data_raw != b"[DONE]":
        try:
            obj = _json.loads(data_raw)
        except Exception:
            obj = None

    if not isinstance(obj, dict):
        # A named `event: error` with a non-JSON/empty body is still
        # structurally an error event (rare, but don't silently swallow it).
        if is_named_error_event:
            return {"kind": "generic_error", "raw_type": None, "message": "", "code": None}
        return None

    top_type = obj.get("type")
    err_obj = obj.get("error") if isinstance(obj.get("error"), dict) else None

    # Structural gate: only proceed if this frame is actually shaped like an
    # error (named error event, top-level type=="error", or an "error" key).
    # A content_block_delta / text / tool-call-argument frame never matches
    # this shape, so it falls through untouched no matter its text content.
    if not is_named_error_event and top_type != "error" and err_obj is None:
        return None

    err_type = ""
    err_message = ""
    err_code = None
    if err_obj:
        err_type = str(err_obj.get("type") or "").lower()
        err_message = str(err_obj.get("message") or "")
        err_code = err_obj.get("code")
    elif top_type == "error":
        err_message = str(obj.get("message") or "")

    hay = (err_type + " " + err_message).lower()

    def _has(*words: str) -> bool:
        return any(w in hay for w in words)

    if _has(
        "context_length", "context length", "too many tokens",
        "maximum context", "prompt is too long", "input length",
        "token limit", "context_length_exceeded",
    ):
        kind = "context_length"
    elif _has("rate_limit", "rate limit", "tpm", "rpm", "too many requests"):
        kind = "rate_limit"
    elif _has("concurrent", "concurrency", "max_concurrent"):
        kind = "concurrency"
    elif _has("idle timeout", "upstream idle", "idle_timeout"):
        kind = "idle_timeout"
    elif _has(
        "overloaded", "resourceexhausted", "resource_exhausted",
        "capacity", "unavailable", "server_error",
    ):
        kind = "overloaded"
    else:
        kind = "generic_error"

    return {"kind": kind, "raw_type": err_type or top_type, "message": err_message, "code": err_code}


def _is_content_sse_frame(frame: bytes) -> bool:
    """
    Structurally determine whether an SSE frame carries *visible* content
    for the Anthropic /messages protocol.

    Unlike the previous substring approach (``b'"text"' in frame`` or
    ``b'content_block_delta' in frame``), this parses the frame's data:
    JSON and inspects only the ``type`` / ``delta.type`` fields.  A model
    that happens to mention the word "text" inside a tool_use input, a
    thinking block, or any other non-content-bearing event will not produce
    a false positive here.

    Returns True for:
      - content_block_delta with delta.type == "text_delta" or
        delta.type == "input_json_delta" (visible assistant content)
      - content_block_start with any content_block.type that carries
        visible output (text, tool_use, etc.)
      - message_start (the assistant has begun emitting, so the stream
        produced something meaningful even if no delta arrived)
    """
    if b"data:" not in frame:
        return False
    data_raw = None
    for line in frame.split(b"\n"):
        line = line.strip(b"\r")
        if line.startswith(b"data:"):
            data_raw = line[5:].strip()
            break
    if not data_raw or data_raw == b"[DONE]":
        return False
    try:
        obj = _json.loads(data_raw)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    etype = obj.get("type", "")
    if etype == "content_block_delta":
        delta = obj.get("delta") or {}
        dt = delta.get("type", "")
        # text_delta carries visible text; input_json_delta carries
        # partial tool-call arguments — both are assistant content.
        return dt in ("text_delta", "input_json_delta")
    if etype == "content_block_start":
        block = obj.get("content_block") or {}
        bt = block.get("type", "")
        # text and tool_use blocks both represent assistant output
        return bt in ("text", "tool_use")
    if etype == "message_start":
        return True
    return False


# Maps mid-stream SSE error kinds (from ``_classify_sse_frame``) to HTTP
# status codes used for key-pool health bookkeeping (``mark_error``).
# Hoisted to module level — was previously redefined inline on every
# mid-stream error frame in both ``event_generator`` and
# ``frame_generator``.
_KIND_TO_STATUS: dict[str, int] = {
    "rate_limit": 429,
    "concurrency": 429,
    "idle_timeout": 408,
    "overloaded": 503,
    "context_length": 400,
    "generic_error": 502,
}


class ProxyCore:
    def __init__(self, pool: KeyPool, provider: str = "openrouter") -> None:
        self.pool = pool
        self.provider = provider
        self.client: Optional[httpx.AsyncClient] = None
        self._prewarm_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        # Cap concurrent streams so we stay under Nvidia free-worker limit (~32)
        self._free_sem = asyncio.Semaphore(FREE_MODEL_MAX_CONCURRENT)

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
            http2=True,
            limits=limits,
            timeout=timeout,
            headers={
                "HTTP-Referer": UPSTREAM_REFERER,
                "X-Title": UPSTREAM_TITLE,
            },
            follow_redirects=True,
        )
        # Pre-warming is strictly background work. Never delay proxy readiness
        # or the first real request waiting for an upstream connection.
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
        if self.pool.total == 0:
            log.warning("Skipping TLS prewarm - no keys available yet")
            return

        assert self.client is not None
        key, _, _ = self.pool.next_key()
        models_url = get_models_url()
        try:
            resp = await self.client.get(
                models_url,
                headers={"Authorization": f"Bearer {key}"},
                timeout=10.0,
            )
            await resp.aread()
            log.info("TLS pre-warm complete (status=%s)", resp.status_code)
        except Exception as e:
            log.warning("Pre-warm failed (non-fatal): %s", e)

    async def _prewarm_loop(self) -> None:
        # Let the proxy become ready and accept real traffic first.
        # Pre-warming must never race the first production request.
        await asyncio.sleep(2.0)
        while True:
            await self._prewarm()
            await asyncio.sleep(PREWARM_INTERVAL)

    async def _health_loop(self) -> None:
        """Periodically log key-pool health; recovery is lazy inside next_key."""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            s = self.pool.stats()
            pl.keys_health(s)  # logging-only rendering
            log.info(
                "Key health: total=%d healthy=%d cooling=%d suspended=%d in_flight=%d",
                s["total"],
                s["healthy"],
                s["cooling"],
                s["suspended"],
                s.get("in_flight", 0),
            )

    def _headers(
        self, key: str, extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": UPSTREAM_REFERER,
            "X-Title": UPSTREAM_TITLE,
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

        # Single authoritative point that overrides any client-supplied model
        # with the active provider's hardcoded default.
        body = _enforce_default_model(body)

        for attempt in range(MAX_RETRIES + 1):
            key, key_idx, is_healthy = self.pool.next_key()
            # Fast-fail when all keys are cooling/suspended — no point burning
            # MAX_RETRIES attempts against keys that will almost certainly fail.
            if not is_healthy and attempt == 0:
                s = self.pool.stats()
                log.warning(
                    "req=%s all keys unhealthy (healthy=%d cooling=%d suspended=%d) — fast-failing with 503",
                    request_id, s["healthy"], s["cooling"], s["suspended"],
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": "All upstream API keys are temporarily unavailable (cooldown/suspended). Try again shortly.",
                            "type": "proxy_error",
                            "code": 503,
                        }
                    },
                    headers={"x-request-id": request_id},
                )
            headers = self._headers(key, extra_headers)
            t0 = time.perf_counter()
            try:
                if stream:
                    result = await self._stream_forward(
                        method, url, headers, body, key_idx, request_id, t0
                    )
                    if isinstance(result, StreamingResponse):
                        return result
                    # Streaming upstream failures normally retry only for
                    # RETRY_STATUSES. Hugging Face has an additional permanent
                    # key-failure path (for example HTTP 402): _stream_forward()
                    # retires that key, so we must also retry here and allow
                    # next_key() to select the replacement key.
                    stream_hf_key_failure = False
                    if self.provider == "huggingface" and result.status_code >= 400:
                        try:
                            error_body = bytes(result.body)
                            stream_hf_key_failure = (
                                is_hf_rate_limit_error(result.status_code, error_body)
                                or is_hf_key_invalid(result.status_code, error_body)
                            )
                        except Exception:
                            stream_hf_key_failure = False

                    if (
                        (
                            result.status_code in RETRY_STATUSES
                            or stream_hf_key_failure
                        )
                        and attempt < MAX_RETRIES
                    ):
                        log.warning(
                            "req=%s key_idx=%d stream-status=%d attempt=%d — retrying with next key",
                            request_id,
                            key_idx,
                            result.status_code,
                            attempt + 1,
                        )
                        last_status = result.status_code
                        continue
                    result.headers["x-request-id"] = request_id
                    return result

                await self.pool.acquire(key_idx)
                pl.trace(request_id).sent(key_idx)  # logging-only lifecycle trace
                try:
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
                        # Check for HF permanent retirement conditions
                        if self.provider == "huggingface":
                            resp_body = resp.content
                            if is_hf_rate_limit_error(status, resp_body) or is_hf_key_invalid(status, resp_body):
                                key_str = self.pool._keys[key_idx].key if key_idx < self.pool._n else ""
                                if key_str:
                                    added, removed = retire_and_remove_hf_key(key_str)
                                    if added:
                                        log.warning(
                                            "req=%s HF key permanently retired (status=%d, key_idx=%d)",
                                            request_id, status, key_idx,
                                        )
                                await self.pool.retire_key(key_idx)
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
                    _tr = pl.trace(request_id)
                    _tr.upstream(key_idx, status)  # logging-only lifecycle trace
                    try:
                        _u = loads(data).get("usage") or {}
                    except Exception:
                        _u = {}
                    _tr.usage(
                        prompt_tokens=_u.get("prompt_tokens"),
                        completion_tokens=_u.get("completion_tokens"),
                        total_tokens=_u.get("total_tokens"),
                    )
                    _tr.finish(status=status)
                    log.info(
                        "req=%s key_idx=%d status=%d latency=%.0fms bytes=%d",
                        request_id,
                        key_idx,
                        status,
                        latency_ms,
                        len(data),
                    )
                    # Enforce response-size cap to bound memory under load
                    if MAX_RESPONSE_BYTES > 0 and len(data) > MAX_RESPONSE_BYTES:
                        await self.pool.mark_error(key_idx, 413)
                        log.warning(
                            "req=%s response too large (%d > %d bytes) — returning 413",
                            request_id, len(data), MAX_RESPONSE_BYTES,
                        )
                        return JSONResponse(
                            status_code=413,
                            content={
                                "error": {
                                    "message": (
                                        f"Response exceeded max size of "
                                        f"{MAX_RESPONSE_BYTES} bytes "
                                        f"(was {len(data)} bytes). "
                                        f"Use stream=True for large responses."
                                    ),
                                    "type": "length",
                                }
                            },
                            headers={"x-request-id": request_id},
                        )
                    return Response(
                        content=data,
                        status_code=status,
                        headers=out_headers,
                        media_type=resp.headers.get("content-type"),
                    )
                finally:
                    await self.pool.release(key_idx)

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

    async def _open_upstream_stream(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes],
        key_idx: int,
        request_id: str,
    ) -> Tuple[Any, int, float] | Response:
        """
        Shared upstream-connection logic for streaming forward paths.

        Acquires the free-model semaphore and pool in-flight slot, builds and
        sends the request, checks the status code.  Returns either:
          - (upstream_response, key_idx, t0) on success (caller must handle
            cleanup in its own finally block: aclose upstream, release key_idx,
            release _free_sem, mark_success/error)
          - Response(...) on connection failure or non-retryable/transient status
            (in this case the caller must NOT do cleanup — this method already
            released everything and returned a terminal Response.)

        This eliminates the duplicated connection + status-check + mark_error
        code between _stream_forward (proxy.py) and translate_stream (routes.py).
        """
        assert self.client is not None
        t0 = time.perf_counter()
        await self._free_sem.acquire()
        await self.pool.acquire(key_idx)
        # Authoritative model override at the boundary to upstream.
        body = _enforce_default_model(body)
        req = self.client.build_request(method, url, headers=headers, content=body)
        try:
            upstream = await self.client.send(req, stream=True)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            await self.pool.mark_error(key_idx, 599)
            await self.pool.release(key_idx)
            self._free_sem.release()
            log.warning(
                "req=%s key_idx=%d stream connect error: %s",
                request_id, key_idx, e,
            )
            return Response(
                content=dumps({
                    "error": {
                        "message": f"upstream connect error: {e}",
                        "type": "proxy_error",
                    }
                }),
                status_code=502,
                media_type="application/json",
            )

        status = upstream.status_code
        if status in RETRY_STATUSES or status >= 400:
            await self.pool.mark_error(key_idx, status)
            # Check for HF permanent retirement conditions (non-streaming pre-response)
            pl.trace(request_id).upstream(key_idx, status)  # logging-only
            if self.provider == "huggingface":
                data = await upstream.aread()
                if is_hf_rate_limit_error(status, data) or is_hf_key_invalid(status, data):
                    key_str = self.pool._keys[key_idx].key if key_idx < self.pool._n else ""
                    if key_str:
                        added, removed = retire_and_remove_hf_key(key_str)
                        if added:
                            log.warning(
                                "req=%s HF key permanently retired in stream (status=%d, key_idx=%d)",
                                request_id, status, key_idx,
                            )
                    await self.pool.retire_key(key_idx)
                await upstream.aclose()
                await self.pool.release(key_idx)
                self._free_sem.release()
                return Response(
                    content=data,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get(
                        "content-type", "application/json"
                    ),
                )
            data = await upstream.aread()
            await upstream.aclose()
            await self.pool.release(key_idx)
            self._free_sem.release()
            return Response(
                content=data,
                status_code=upstream.status_code,
                media_type=upstream.headers.get(
                    "content-type", "application/json"
                ),
            )
        log.info(
            "req=%s key_idx=%d status=%d stream=1",
            request_id, key_idx, status,
        )
        pl.trace(request_id).upstream(key_idx, status)  # logging-only lifecycle trace
        return upstream, key_idx, t0

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
        result = await self._open_upstream_stream(
            method, url, headers, body, key_idx, request_id
        )
        if isinstance(result, Response):
            return result
        upstream, key_idx, t0 = result

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

        async def event_generator() -> AsyncIterator[bytes]:
            """
            Pass-through SSE with:
              • drop OpenAI-style data: [DONE] trailers
              • detect mid-stream provider errors (rate limit / concurrent / idle)
              • emit SSE keepalives so clients/middleboxes do not idle-out
              • Anthropic: synthesize message_stop ONLY on clean early close
                that already produced content (never on provider overload)
              • track whether any bytes were yielded
              • always release the key's in-flight slot + free-model semaphore
            """

            buf = b""
            saw_message_stop = False
            stream_error = False
            stream_error_reason = ""  # diagnostic: who caused the stream to fail
            saw_content = False
            yielded_any = False
            is_messages = "/messages" in url
            try:
                keepalive_interval = float(PROXY_KEEPALIVE_SECONDS) if PROXY_KEEPALIVE_SECONDS else 15.0
            except NameError:
                keepalive_interval = 15.0

            try:
                aiter = upstream.aiter_raw()
                while True:
                    try:
                        raw = await asyncio.wait_for(aiter.__anext__(), timeout=keepalive_interval)
                    except asyncio.TimeoutError:
                        # No data from upstream → emit client keepalive and keep waiting
                        if not stream_error:
                            yield b": keepalive\n\n"
                            yielded_any = True
                        continue
                    except StopAsyncIteration:
                        break

                    if not raw:
                        continue
                    buf += raw

                    while True:
                        sep = buf.find(b"\n\n")
                        crlf = False
                        if sep < 0:
                            sep = buf.find(b"\r\n\r\n")
                            if sep < 0:
                                break
                            crlf = True
                        frame = buf[:sep]
                        buf = buf[sep + (4 if crlf else 2):]

                        if is_openai_done_frame(frame):
                            continue

                        frame_error = _classify_sse_frame(frame)
                        if frame_error is not None:
                            stream_error = True
                            stream_error_reason = f"provider_error/{frame_error['kind']}"
                            log.warning(
                                "req=%s key_idx=%d mid-stream provider error kind=%s type=%s: %s",
                                request_id,
                                key_idx,
                                frame_error["kind"],
                                frame_error.get("raw_type"),
                                frame[:300],
                            )
                            # Map the *structured* error kind to a status.
                            # Never collapse everything to 429: context-length
                            # errors are not rate limits, and overload/idle
                            # are distinct upstream conditions.
                            err_status = _KIND_TO_STATUS.get(frame_error["kind"], 502)
                            try:
                                await self.pool.mark_error(key_idx, err_status)
                            except Exception:
                                pass
                            # HF streaming rate-limit → permanent retirement
                            if self.provider == "huggingface" and frame_error["kind"] in (
                                "rate_limit", "concurrency"
                            ):
                                key_str = self.pool._keys[key_idx].key if key_idx < self.pool._n else ""
                                if key_str:
                                    retire_and_remove_hf_key(key_str)
                                await self.pool.retire_key(key_idx)

                            # Do NOT yield the raw error frame to the client — it
                            # confuses Anthropic/OpenAI SDKs (they parse it as a
                            # valid chunk and then mismatch final usage). Just mark
                            # the key unhealthy and return; the finally block
                            # synthesizes message_stop for is_messages streams so
                            # the Anthropic client doesn't hang in "Thought for…".
                            # The next request will rotate off this key via
                            # next_key()'s cooldown skip.
                            return

                        if is_messages and b"message_stop" in frame:
                            saw_message_stop = True
                        if _is_content_sse_frame(frame):
                            saw_content = True
                        yield frame + b"\n\n"
                        yielded_any = True

                if buf.strip() and not is_openai_done_frame(buf):
                    if _is_content_sse_frame(buf):
                        saw_content = True
                    if is_messages and b"message_stop" in buf:
                        saw_message_stop = True
                    yield buf if buf.endswith(b"\n\n") else buf + b"\n\n"
                    yielded_any = True

            except (httpx.ReadError, httpx.StreamError) as e:
                stream_error = True
                stream_error_reason = f"upstream_closed/{type(e).__name__}: {e}"
                log.warning(
                    "req=%s key_idx=%d stream upstream closed early: %s (saw_message_stop=%s yielded=%s)",
                    request_id,
                    key_idx,
                    e,
                    saw_message_stop,
                    yielded_any,
                )
                try:
                    await self.pool.mark_error(key_idx, 599)
                except Exception:
                    pass
            except asyncio.CancelledError:
                stream_error = True
                stream_error_reason = "client_cancelled"
                log.info("req=%s stream cancelled by client", request_id)
                raise
            except Exception as e:
                stream_error = True
                stream_error_reason = f"mid_stream_error/{type(e).__name__}: {e}"
                log.warning("req=%s stream mid-body error: %s", request_id, e)
                try:
                    await self.pool.mark_error(key_idx, 500)
                except Exception:
                    pass
            finally:
                # For is_messages (Anthropic-format) streams, always terminate
                # with a clean message_stop if we haven't already. This applies
                # to both the "incomplete upstream" case (saw_content but no
                # message_stop) AND mid-stream error cases (stream_error=True)
                # so the Anthropic SDK doesn't hang in "Thought for…" waiting
                # for a stop frame after a provider_error (e.g. 402 Insufficient
                # balance) cuts the stream short.
                if (
                    is_messages
                    and saw_content
                    and not saw_message_stop
                ):
                    log.warning(
                        "req=%s synthesizing message_stop after incomplete upstream (stream_error=%s reason=%s)",
                        request_id,
                        stream_error,
                        stream_error_reason,
                    )
                    tail = (
                        b'event: message_delta\n'
                        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                        b'"usage":{"output_tokens":0}}\n\n'
                        b'event: message_stop\n'
                        b'data: {"type":"message_stop"}\n\n'
                    )
                    try:
                        yield tail
                    except Exception:
                        pass
                elif is_messages and not saw_message_stop and stream_error:
                    log.warning(
                        "req=%s skipping message_stop synthesis (stream_error=True yielded=%s reason=%s)",
                        request_id,
                        yielded_any,
                        stream_error_reason,
                    )
                    # If the client never received any content, surface a clean
                    # Anthropic error so Claude Code does not hang in "Thought for …".
                    if not yielded_any:
                        try:
                            err = (
                                b'event: error\n'
                                b'data: {"type":"error","error":{"type":"api_error",'
                                b'"message":"Upstream idle timeout with no content - retry the turn"}}\n\n'
                            )
                            yield err
                        except Exception:
                            pass

                if not stream_error:
                    await _quiet_cleanup(
                        self.pool.mark_success,
                        key_idx, (time.perf_counter() - t0) * 1000,
                        what="mark_success", rid=request_id,
                    )

                # logging-only lifecycle completion
                _tr = pl.trace(request_id)
                if not stream_error:
                    _tr.finish(status=200)
                else:
                    _tr.fail(status=599, phase="upstream", error=f"upstream closed early / mid-stream error ({stream_error_reason})")

                await _quiet_cleanup(upstream.aclose, what="upstream.aclose", rid=request_id)
                await _quiet_cleanup(self.pool.release, key_idx, what="pool.release", rid=request_id)
                await _quiet_cleanup(self._free_sem.release, what="free_sem.release", rid=request_id)

        return StreamingResponse(
            event_generator(),
            status_code=upstream.status_code,
            headers=out_headers,
            media_type="text/event-stream",
        )

    async def iter_upstream_sse(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes],
        key_idx: int,
        request_id: str,
    ) -> AsyncIterator[bytes] | Response:
        """
        Shared upstream SSE frame iterator for translation paths.

        Handles:
        - Connection acquisition (_free_sem, pool acquire)
        - Upstream request send
        - Status code check with mark_error/release on failure
        - SSE frame iteration with keepalive emission
        - Mid-stream provider error classification via _classify_sse_frame
        - Cleanup on exit (aclose, release, semaphore release, mark_success/error)

        Returns an AsyncIterator yielding raw SSE frames (bytes) on success,
        or a Response object on connection/upstream failure (caller must return it directly).
        """
        assert self.client is not None
        t0 = time.perf_counter()
        await self._free_sem.acquire()
        await self.pool.acquire(key_idx)
        # Authoritative model override at the boundary to upstream.
        body = _enforce_default_model(body)
        req = self.client.build_request(method, url, headers=headers, content=body)
        try:
            upstream = await self.client.send(req, stream=True)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            await self.pool.mark_error(key_idx, 599)
            await self.pool.release(key_idx)
            self._free_sem.release()
            log.warning(
                "req=%s key_idx=%d stream connect error: %s",
                request_id, key_idx, e,
            )
            return Response(
                content=dumps({
                    "error": {
                        "message": f"upstream connect error: {e}",
                        "type": "proxy_error",
                    }
                }),
                status_code=502,
                media_type="application/json",
            )

        status = upstream.status_code
        if status in RETRY_STATUSES or status >= 400:
            await self.pool.mark_error(key_idx, status)
            data = await upstream.aread()
            await upstream.aclose()
            await self.pool.release(key_idx)
            self._free_sem.release()
            return Response(
                content=data,
                status_code=upstream.status_code,
                media_type=upstream.headers.get(
                    "content-type", "application/json"
                ),
            )

        log.info(
            "req=%s key_idx=%d status=%d stream=1",
            request_id, key_idx, status,
        )
        pl.trace(request_id).upstream(key_idx, status)  # logging-only lifecycle trace

        async def frame_generator() -> AsyncIterator[bytes]:
            """
            Yield raw SSE frames with keepalive and error detection.
            """

            buf = b""
            stream_error = False
            stream_error_reason = ""  # diagnostic: who caused the stream to fail
            try:
                keepalive_interval = float(PROXY_KEEPALIVE_SECONDS) if PROXY_KEEPALIVE_SECONDS else 15.0
            except NameError:
                keepalive_interval = 15.0

            try:
                aiter = upstream.aiter_raw()
                while True:
                    try:
                        raw = await asyncio.wait_for(aiter.__anext__(), timeout=keepalive_interval)
                    except asyncio.TimeoutError:
                        # No data from upstream → emit client keepalive and keep waiting
                        if not stream_error:
                            yield b": keepalive\n\n"
                        continue
                    except StopAsyncIteration:
                        break

                    if not raw:
                        continue
                    buf += raw

                    while True:
                        sep = buf.find(b"\n\n")
                        crlf = False
                        if sep < 0:
                            sep = buf.find(b"\r\n\r\n")
                            if sep < 0:
                                break
                            crlf = True
                        frame = buf[:sep]
                        buf = buf[sep + (4 if crlf else 2):]

                        if is_openai_done_frame(frame):
                            continue

                        frame_error = _classify_sse_frame(frame)
                        if frame_error is not None:
                            stream_error = True
                            stream_error_reason = f"provider_error/{frame_error['kind']}"
                            log.warning(
                                "req=%s key_idx=%d mid-stream provider error kind=%s type=%s: %s",
                                request_id,
                                key_idx,
                                frame_error["kind"],
                                frame_error.get("raw_type"),
                                frame[:300],
                            )
                            # Map the structured error kind to a status
                            err_status = _KIND_TO_STATUS.get(frame_error["kind"], 502)
                            try:
                                await self.pool.mark_error(key_idx, err_status)
                            except Exception:
                                pass
                            # HF streaming rate-limit → permanent retirement
                            if self.provider == "huggingface" and frame_error["kind"] in (
                                "rate_limit", "concurrency"
                            ):
                                key_str = self.pool._keys[key_idx].key if key_idx < self.pool._n else ""
                                if key_str:
                                    retire_and_remove_hf_key(key_str)
                                await self.pool.retire_key(key_idx)

                            # Do NOT yield the error frame — terminate the stream
                            # cleanly so the next request rotates off this key via
                            # next_key()'s cooldown skip. Clients see a clean SSE
                            # EOF (Anthropic clients will need a message_stop —
                            # the finally block handles that).
                            return

                        yield frame + b"\n\n"

                if buf.strip() and not is_openai_done_frame(buf):
                    yield buf if buf.endswith(b"\n\n") else buf + b"\n\n"

            except (httpx.ReadError, httpx.StreamError) as e:
                stream_error = True
                stream_error_reason = f"upstream_closed/{type(e).__name__}: {e}"
                log.warning(
                    "req=%s key_idx=%d stream upstream closed early: %s",
                    request_id,
                    key_idx,
                    e,
                )
                try:
                    await self.pool.mark_error(key_idx, 599)
                except Exception:
                    pass
            except asyncio.CancelledError:
                stream_error = True
                stream_error_reason = "client_cancelled"
                log.info("req=%s stream cancelled by client", request_id)
                raise
            except Exception as e:
                stream_error = True
                stream_error_reason = f"mid_stream_error/{type(e).__name__}: {e}"
                log.warning("req=%s stream mid-body error: %s", request_id, e)
                try:
                    await self.pool.mark_error(key_idx, 500)
                except Exception:
                    pass
            finally:
                if not stream_error:
                    await _quiet_cleanup(
                        self.pool.mark_success,
                        key_idx, (time.perf_counter() - t0) * 1000,
                        what="mark_success", rid=request_id,
                    )

                # logging-only lifecycle completion
                _tr = pl.trace(request_id)
                if not stream_error:
                    _tr.finish(status=200)
                else:
                    _tr.fail(status=599, phase="upstream", error=f"upstream closed early / mid-stream error ({stream_error_reason})")

                await _quiet_cleanup(upstream.aclose, what="upstream.aclose", rid=request_id)
                await _quiet_cleanup(self.pool.release, key_idx, what="pool.release", rid=request_id)
                await _quiet_cleanup(self._free_sem.release, what="free_sem.release", rid=request_id)

        return frame_generator()
