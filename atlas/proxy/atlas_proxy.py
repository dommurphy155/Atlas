from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from proxy.nvidia_key_store import NvidiaKeyStore
from proxy.nvidia_client import NvidiaClient
from proxy.openai_compat import (
    anthropic_openai_payload,
    anthropic_response_from_blocks,
    anthropic_sse_from_response,
    normalize_messages,
    openai_response_from_router,
    openai_response_to_anthropic,
    openai_error,
    openai_sse_to_anthropic_sse,
    sse_from_text,
)
from proxy.stats import record_failure, record_success, get_status as stats_status
from proxy.system_prompt import replace_system_prompt

MAX_BODY_BYTES = 2 * 1024 * 1024

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# Atlas is a single-provider NVIDIA proxy. Every request routes directly to
# NVIDIA's chat-completions endpoint. No fallback, no provider switching.
HOST = os.getenv("ATLAS_PROXY_HOST", "127.0.0.1")
PORT = int(os.getenv("ATLAS_PROXY_PORT", "8788"))
KEYS_FILE = os.getenv("ATLAS_KEYS_FILE", str(ROOT_DIR / "data" / "keys.txt"))
NVIDIA_MODEL = os.getenv("ATLAS_NVIDIA_MODEL", "z-ai/glm-5.2")
NVIDIA_BASE_URL = os.getenv("ATLAS_NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
RELOAD_SECONDS = int(os.getenv("ATLAS_PROXY_RELOAD_SECONDS", "5"))
REQUEST_TIMEOUT = float(os.getenv("ATLAS_PROXY_REQUEST_TIMEOUT", "300"))
CONNECT_TIMEOUT = float(os.getenv("ATLAS_PROXY_CONNECT_TIMEOUT", "10"))
# Stream read deadline — the dead-stream backstop, NOT the thinking-gap limit.
# Reasoning models sit silent for long stretches; 60s killed healthy streams.
# 180s gives a genuinely dead upstream time to be detected without murdering
# a model that's just thinking. The keepalive wrapper keeps the downstream
# client alive well before this fires.
READ_TIMEOUT = float(os.getenv("ATLAS_PROXY_READ_TIMEOUT", "180"))
# SSE keepalive cadence (seconds). While the upstream is silent, the proxy
# emits ': keepalive\n\n' comment lines so downstream clients and any
# middleboxes (nginx proxy_read_timeout, corporate proxies) reset their idle
# timers instead of killing a healthy-but-quiet stream.
KEEPALIVE_SECONDS = float(os.getenv("ATLAS_PROXY_KEEPALIVE_SECONDS", "15"))
MAX_RETRIES = int(os.getenv("ATLAS_PROXY_MAX_RETRIES", "2"))
MAX_KEY_FAILOVERS = int(os.getenv("ATLAS_PROXY_MAX_KEY_FAILOVERS", "3"))
DEBUG = os.getenv("ATLAS_PROXY_DEBUG", "0") == "1"


class _CleanFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%d/%-m/%y %-H:%M")
        color = self._COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:<7}{self._RESET}"
        return f"{ts} {level} atlas   {record.getMessage()}"


_handler = logging.StreamHandler()
_handler.setFormatter(_CleanFormatter())
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    handlers=[_handler],
    force=True,
)


def _short_model(name: str) -> str:
    """Shorten model name for log display: moonshotai/kimi-k2.6 -> kimi-k2.6"""
    if "/" in name:
        return name.rsplit("/", 1)[1]
    return name


logger = logging.getLogger("atlas-proxy")

# Silence everything that's not our logger
for _name in ("uvicorn", "uvicorn.access", "uvicorn.error", "httpx", "httpx._client", "watchfiles"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

key_store = NvidiaKeyStore(KEYS_FILE, RELOAD_SECONDS)
nvidia_client = NvidiaClient(NVIDIA_BASE_URL, REQUEST_TIMEOUT, CONNECT_TIMEOUT, READ_TIMEOUT)
watch_task: asyncio.Task[None] | None = None
active_requests = 0


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global watch_task
    await key_store.load(force=True)
    watch_task = asyncio.create_task(key_store.watch())
    logger.info("atlas started on %s:%s using keys_file=%s model=%s", HOST, PORT, KEYS_FILE, NVIDIA_MODEL)
    try:
        yield
    finally:
        if watch_task:
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass
        await nvidia_client.close()


app = FastAPI(title="Atlas Proxy", lifespan=lifespan)


def json_error(message: str, code: str, status: int) -> JSONResponse:
    return JSONResponse(openai_error(message, code, status), status_code=status)


def _generate_rid() -> str:
    """Generate a short request ID for tracing."""
    return uuid.uuid4().hex[:8]


def _extract_usage(json_data: dict[str, Any]) -> tuple[int, int, int, int]:
    """Extract (prompt_tokens, completion_tokens, total_tokens, tool_calls)."""
    usage = json_data.get("usage") if isinstance(json_data, dict) else None
    if not isinstance(usage, dict):
        return 0, 0, 0, 0
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    tt = int(usage.get("total_tokens") or 0)
    tc = 0
    for choice in (json_data.get("choices") or []):
        if isinstance(choice, dict):
            msg = choice.get("message")
            if isinstance(msg, dict):
                for item in (msg.get("tool_calls") or []):
                    if isinstance(item, dict):
                        tc += 1
    return pt, ct, tt, tc


async def parse_request_body(request: Request) -> dict[str, Any] | JSONResponse:
    size = request.headers.get("content-length")
    if size:
        try:
            if int(size) > MAX_BODY_BYTES:
                return json_error("request body too large", "request_too_large", 413)
        except ValueError:
            return json_error("invalid content-length", "bad_request", 400)

    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        return json_error("request body too large", "request_too_large", 413)

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return json_error("bad json", "bad_json", 400)

    if not isinstance(payload, dict):
        return json_error("json body must be an object", "bad_json", 400)
    return payload


def upstream_error_text(response: Any) -> str:
    if response.json_data:
        return json.dumps(response.json_data)
    return response.text or "upstream error"


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "atlas-proxy",
        "provider": "nvidia",
        "model": NVIDIA_MODEL,
        "host": HOST,
        "port": PORT,
        "keys_available": key_store.available,
    }


@app.get("/stats")
async def stats() -> dict[str, Any]:
    nvidia_stats = key_store.stats()
    proxy_stats = stats_status()
    return {
        "status": "ok",
        "provider": "nvidia",
        "model": NVIDIA_MODEL,
        "nvidia_base_url": NVIDIA_BASE_URL,
        "keys_file": KEYS_FILE,
        "nvidia_keys_total": nvidia_stats["total_keys"],
        "nvidia_keys_available": nvidia_stats["available"],
        "nvidia_keys_cooling_down": nvidia_stats["cooling_down"],
        "active_requests": active_requests,
        "proxy": proxy_stats,
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": NVIDIA_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "nvidia",
            }
        ],
    }


# ── Request endpoints ──────────────────────────────────────────────────


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    global active_requests
    body = await parse_request_body(request)
    if isinstance(body, JSONResponse):
        return body

    try:
        messages = normalize_messages(body.get("messages"))
    except ValueError as exc:
        return json_error(str(exc), "bad_request", 400)

    model = str(body.get("model") or NVIDIA_MODEL)
    if model == "default":
        model = NVIDIA_MODEL

    upstream_payload = dict(body)
    upstream_payload["model"] = model
    upstream_payload["messages"] = messages
    stream = bool(upstream_payload.get("stream", False))

    replace_system_prompt(upstream_payload, provider="openai")

    rid = _generate_rid()
    started = time.monotonic()
    logger.info(">%s %s stream=%s provider=nvidia", rid, _short_model(model), "yes" if stream else "no")

    if stream:
        # Increment here; the matching decrement happens in
        # stream_with_active_count when the stream body finishes. Don't
        # decrement on return — the body hasn't been consumed yet.
        active_requests += 1
        try:
            response = await handle_stream(model, upstream_payload, rid, started)
            return response
        except Exception:
            active_requests -= 1
            elapsed = time.monotonic() - started
            logger.info("<%s status=500 provider=nvidia %.1fs", rid, elapsed)
            record_failure("nvidia")
            raise

    active_requests += 1
    try:
        result = await handle_non_stream(model, upstream_payload, rid, started)
        elapsed = time.monotonic() - started
        if result.status_code >= 400:
            logger.info("<%s status=%d provider=nvidia model=%s %.1fs", rid, result.status_code, _short_model(model), elapsed)
        return result
    finally:
        active_requests -= 1


@app.post("/v1/messages", response_model=None)
async def anthropic_messages(request: Request) -> JSONResponse | StreamingResponse:
    global active_requests
    body = await parse_request_body(request)
    if isinstance(body, JSONResponse):
        return body

    requested_model = str(body.get("model") or "claude")
    replace_system_prompt(body, provider="anthropic")

    try:
        payload = anthropic_openai_payload(body, NVIDIA_MODEL)
    except ValueError as exc:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}},
            status_code=400,
        )

    rid = _generate_rid()
    started = time.monotonic()

    # Streaming: route through the real-time OpenAI→Anthropic SSE adapter
    # instead of buffering the whole response and faking the event stream.
    # Increment here; stream_with_active_count decrements when the body
    # finishes. handle_anthropic_stream always returns a StreamingResponse
    # (success and error paths are both wrapped in stream_with_active_count),
    # so the single decrement in the wrapper is the matching one.
    if body.get("stream"):
        active_requests += 1
        response = await handle_anthropic_stream(requested_model, payload, rid, started)
        elapsed = time.monotonic() - started
        if isinstance(response, StreamingResponse):
            logger.info("<%s status=200 stream=yes provider=nvidia model=%s %.1fs", rid, _short_model(NVIDIA_MODEL), elapsed)
        else:
            logger.info("<%s status=%d stream=yes-failed provider=nvidia %.1fs", rid, getattr(response, "status_code", 0), elapsed)
        return response

    active_requests += 1
    try:
        response = await handle_non_stream(NVIDIA_MODEL, payload, rid, started)
    finally:
        active_requests -= 1

    elapsed = time.monotonic() - started
    logger.info("<%s status=%d %.1fs", rid, response.status_code, elapsed)

    if response.status_code != 200:
        data = json.loads(response.body.decode("utf-8"))
        message = data.get("error", {}).get("message", "proxy error")
        status = response.status_code
        record_failure("nvidia")
        if body.get("stream"):
            error_response = anthropic_response_from_blocks(
                requested_model,
                [{"type": "text", "text": f"Atlas proxy error ({status}): {message}"}],
                "error",
            )
            return StreamingResponse(
                anthropic_sse_from_response(error_response),
                status_code=200 if status < 500 else status,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": str(message)}},
            status_code=status,
        )

    openai_payload = json.loads(response.body.decode("utf-8"))
    pt, ct, tt, tc = _extract_usage(openai_payload)
    record_success("nvidia", NVIDIA_MODEL, pt, ct, tt, tc)

    anthropic_payload = openai_response_to_anthropic(requested_model, openai_payload)

    if body.get("stream"):
        return StreamingResponse(
            anthropic_sse_from_response(anthropic_payload),
            status_code=200,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse(anthropic_payload)


# ── Handler: non-streaming ────────────────────────────────────────────


async def handle_non_stream(
    model: str,
    payload: dict[str, Any],
    rid: str = "",
    started: float = 0.0,
) -> JSONResponse:
    last_status = 503
    last_message = "no usable NVIDIA keys are available"
    key_failovers = 0
    server_retries = 0

    for _ in range(max(1, MAX_KEY_FAILOVERS + MAX_RETRIES + 1)):
        if key_failovers >= MAX_KEY_FAILOVERS:
            record_failure("nvidia")
            return json_error(
                "Atlas exhausted the per-request key failover limit. Retry request.",
                "key_failover_limit",
                503,
            )
        api_key = await key_store.acquire()
        if not api_key:
            record_failure("nvidia")
            return json_error("no usable NVIDIA keys are available", "no_usable_keys", 503)

        try:
            response = await nvidia_client.chat(api_key, payload)
        except httpx.TimeoutException:
            await key_store.cooldown_key(api_key)
            record_failure("nvidia")
            return json_error("upstream request timed out", "upstream_timeout", 504)
        except httpx.HTTPError as exc:
            await key_store.cooldown_key(api_key)
            last_status = 502
            last_message = f"upstream request failed: {exc.__class__.__name__}"
            key_failovers += 1
            continue

        if response.status_code < 400 and response.json_data is not None:
            pt, ct, tt, tc = _extract_usage(response.json_data)
            record_success("nvidia", model, pt, ct, tt, tc)
            elapsed = time.monotonic() - started if started else 0
            logger.info(
                "<%s status=%d provider=nvidia model=%s tools=%d in_tokens=%d out_tokens=%d Total=%d %.1fs",
                rid, response.status_code, _short_model(model), tc, pt, ct, tt, elapsed,
            )
            return JSONResponse(openai_response_from_router(model, response.json_data))

        last_status = response.status_code
        last_message = upstream_error_text(response)

        # Rate-limited / quota — cool the key and try another.
        if response.status_code in {402, 429}:
            reason = "credits exhausted" if response.status_code == 402 else "quota/billing 429"
            await key_store.cooldown_key(api_key)
            logger.warning("nvidia key cooled (%s) after %d", reason, response.status_code)
            key_failovers += 1
            continue

        # Invalid / forbidden — cool the key and try another.
        if response.status_code in {401, 403}:
            await key_store.cooldown_key(api_key)
            logger.warning("nvidia key invalid/forbidden after %d", response.status_code)
            key_failovers += 1
            continue

        # Transient upstream — small retry on the same key pool.
        if response.status_code in {500, 502, 503, 504}:
            if server_retries >= MAX_RETRIES:
                break
            server_retries += 1
            continue

        # Any other 4xx — surface it, no point retrying.
        break

    record_failure("nvidia")
    return json_error(last_message, "upstream_error", last_status)


# ── Handler: streaming ────────────────────────────────────────────────


async def handle_stream(
    model: str,
    payload: dict[str, Any],
    rid: str = "",
    started: float = 0.0,
) -> StreamingResponse | JSONResponse:
    key_failovers = 0
    server_retries = 0

    for _ in range(max(1, MAX_KEY_FAILOVERS + MAX_RETRIES + 1)):
        if key_failovers >= MAX_KEY_FAILOVERS:
            record_failure("nvidia")
            return stream_error(model, "Atlas exhausted the per-request key failover limit. Retry request.", 503)
        api_key = await key_store.acquire()
        if not api_key:
            record_failure("nvidia")
            return stream_error(model, "no usable NVIDIA keys are available", 503)

        def _on_timeout() -> None:
            # Mid-stream timeout: cool the key and record the failure. Without
            # this, a key that returns 200-then-hang recycles with no cooldown
            # and the death is counted as a success in /stats.
            asyncio.create_task(key_store.cooldown_key(api_key))
            record_failure("nvidia")

        try:
            status, _, iterator = await nvidia_client.stream_chat(
                api_key, payload, rid=rid, on_timeout=_on_timeout
            )
        except httpx.TimeoutException:
            await key_store.cooldown_key(api_key)
            record_failure("nvidia")
            return stream_error(model, "upstream request timed out", 504)
        except httpx.HTTPError as exc:
            await key_store.cooldown_key(api_key)
            logger.warning("stream request failed: %s", exc.__class__.__name__)
            key_failovers += 1
            continue

        if status < 400:
            return StreamingResponse(
                stream_with_active_count(
                    keepalive(
                        stream_router_sse(iterator, model, rid, "nvidia", started),
                        KEEPALIVE_SECONDS,
                    )
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        if status in {402, 429, 401, 403}:
            await key_store.cooldown_key(api_key)
            logger.warning("nvidia key cooled after stream %d", status)
            key_failovers += 1
            continue

        if status in {500, 502, 503, 504}:
            if server_retries >= MAX_RETRIES:
                record_failure("nvidia")
                return stream_error(model, f"upstream returned {status}", status)
            server_retries += 1
            continue

        record_failure("nvidia")
        return stream_error(model, f"upstream returned {status}", status)

    record_failure("nvidia")
    return stream_error(model, "no usable NVIDIA keys are available", 503)


# ── Handler: Anthropic streaming ───────────────────────────────────────


def _anthropic_stream_error(model: str, message: str, status: int) -> StreamingResponse:
    """Emit an Anthropic-shaped SSE error stream (for /v1/messages stream failures)."""
    error_response = anthropic_response_from_blocks(
        model,
        [{"type": "text", "text": f"Atlas proxy error ({status}): {message}"}],
        "error",
    )

    async def iterator() -> AsyncIterator[bytes]:
        async for chunk in anthropic_sse_from_response(error_response):
            yield chunk

    return StreamingResponse(
        stream_with_active_count(iterator()),
        status_code=200 if status < 500 else status,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def handle_anthropic_stream(
    requested_model: str,
    payload: dict[str, Any],
    rid: str = "",
    started: float = 0.0,
) -> StreamingResponse:
    """Stream /v1/messages by translating NVIDIA's OpenAI SSE into Anthropic SSE.

    Same key-failover/retry loop as handle_stream(), but the upstream iterator
    is wrapped in openai_sse_to_anthropic_sse() and errors are Anthropic-shaped.
    """
    key_failovers = 0
    server_retries = 0

    def _on_done(pt: int, ct: int, tt: int, tc: int) -> None:
        record_success("nvidia", NVIDIA_MODEL, pt, ct, tt, tc)
        elapsed = time.monotonic() - started if started else 0
        logger.info(
            "<%s status=200 provider=nvidia model=%s tools=%d in_tokens=%d out_tokens=%d Total=%d %.1fs",
            rid, _short_model(NVIDIA_MODEL), tc, pt, ct, tt, elapsed,
        )

    for _ in range(max(1, MAX_KEY_FAILOVERS + MAX_RETRIES + 1)):
        if key_failovers >= MAX_KEY_FAILOVERS:
            record_failure("nvidia")
            return _anthropic_stream_error(requested_model, "Atlas exhausted the per-request key failover limit. Retry request.", 503)
        api_key = await key_store.acquire()
        if not api_key:
            record_failure("nvidia")
            return _anthropic_stream_error(requested_model, "no usable NVIDIA keys are available", 503)

        def _on_timeout() -> None:
            asyncio.create_task(key_store.cooldown_key(api_key))
            record_failure("nvidia")

        try:
            status, _, iterator = await nvidia_client.stream_chat(
                api_key, payload, rid=rid, on_timeout=_on_timeout
            )
        except httpx.TimeoutException:
            await key_store.cooldown_key(api_key)
            record_failure("nvidia")
            return _anthropic_stream_error(requested_model, "upstream request timed out", 504)
        except httpx.HTTPError as exc:
            await key_store.cooldown_key(api_key)
            logger.warning("anthropic stream request failed: %s", exc.__class__.__name__)
            key_failovers += 1
            continue

        if status < 400:
            return StreamingResponse(
                stream_with_active_count(
                    keepalive(
                        openai_sse_to_anthropic_sse(iterator, requested_model, on_done=_on_done),
                        KEEPALIVE_SECONDS,
                    )
                ),
                status_code=200,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        if status in {402, 429, 401, 403}:
            await key_store.cooldown_key(api_key)
            logger.warning("nvidia key cooled after anthropic stream %d", status)
            key_failovers += 1
            continue

        if status in {500, 502, 503, 504}:
            if server_retries >= MAX_RETRIES:
                record_failure("nvidia")
                return _anthropic_stream_error(requested_model, f"upstream returned {status}", status)
            server_retries += 1
            continue

        record_failure("nvidia")
        return _anthropic_stream_error(requested_model, f"upstream returned {status}", status)

    record_failure("nvidia")
    return _anthropic_stream_error(requested_model, "no usable NVIDIA keys are available", 503)


# ── SSE helpers ───────────────────────────────────────────────────────


async def stream_router_sse(
    iterator: AsyncIterator[bytes],
    model: str,
    rid: str,
    provider: str,
    started: float,
) -> AsyncIterator[bytes]:
    """Stream SSE chunks, capturing the last chunk for usage data, then logging."""
    last_chunk = b""
    async for chunk in iterator:
        last_chunk = chunk
        yield chunk

    # Extract usage from the last SSE data line (contains the [DONE] or final chunk)
    try:
        text = last_chunk.decode("utf-8", errors="replace")
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                    usage = data.get("choices", [{}])[-1].get("usage")
                    if isinstance(usage, dict):
                        tt = int(usage.get("total_tokens", 0))
                        tc = 0
                        for item in (data.get("choices", [{}])[-1].get("delta", {}).get("tool_calls") or []):
                            if isinstance(item, dict):
                                tc += 1
                        if tt == 0:
                            record_success(provider, model, 0, 0, 0, 0)
                        else:
                            pt = int(usage.get("prompt_tokens", 0))
                            ct = int(usage.get("completion_tokens", 0))
                            record_success(provider, model, pt, ct, tt, tc)
                        elapsed = time.monotonic() - started
                        logger.info(
                            "<%s status=200 provider=nvidia model=%s tools=%d in_tokens=%d out_tokens=%d Total=%d %.1fs",
                            rid, _short_model(model), tc, pt, ct, tt, elapsed,
                        )
                        return
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass
        record_success(provider, model, 0, 0, 0, 0)
        elapsed = time.monotonic() - started
        logger.info("<%s status=200 provider=nvidia model=%s tools=? in_tokens=? out_tokens=? %.1fs", rid, _short_model(model), elapsed)
    except Exception:
        record_success(provider, model, 0, 0, 0, 0)
        elapsed = time.monotonic() - started
        logger.info("<%s status=200 provider=nvidia model=%s %.1fs", rid, _short_model(model), elapsed)


async def stream_with_active_count(iterator: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    global active_requests
    try:
        async for chunk in iterator:
            yield chunk
    finally:
        active_requests -= 1


async def keepalive(iterator: AsyncIterator[bytes], interval: float) -> AsyncIterator[bytes]:
    """Emit SSE keepalive comments during upstream idle periods.

    Reasoning models sit silent for long stretches (prefill, thinking). A bare
    idle gap can trip downstream client timeouts and middlebox idle timers
    (nginx ``proxy_read_timeout``, corporate proxies) even when the upstream
    stream is healthy. While NVIDIA is quiet, emit ``: keepalive\\n\\n`` — an
    SSE comment line that conformant clients ignore but that resets every idle
    timer between us and the client.

    Interleaves keepalives with upstream bytes with no buffering, so real
    tokens still stream the instant they arrive. The in-flight upstream read
    is shielded so the keepalive timeout races *against* it without cancelling
    it — cancelling ``__anext__`` would abort the generator's current await
    and drop the chunk it was about to yield. The upstream's own read deadline
    (NvidiaClient stream client) remains the dead-stream backstop.
    """
    keepalive_chunk = b": keepalive\n\n"
    iterator = iterator.__aiter__()
    pending: asyncio.Task[bytes] | None = None
    while True:
        if pending is None:
            pending = asyncio.ensure_future(iterator.__anext__())
        done, _ = await asyncio.wait(
            {pending}, timeout=interval, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            # Interval elapsed with no upstream byte — emit a keepalive and
            # keep waiting on the same in-flight read.
            yield keepalive_chunk
            continue
        try:
            chunk = pending.result()
        except StopAsyncIteration:
            return
        pending = None
        yield chunk


def stream_error(model: str, message: str, status: int) -> StreamingResponse:
    async def iterator() -> AsyncIterator[bytes]:
        async for chunk in sse_from_text(model, f"Atlas proxy error ({status}): {message}"):
            yield chunk

    return StreamingResponse(
        stream_with_active_count(iterator()),
        status_code=status,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Entrypoint ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Atlas NVIDIA OpenAI-compatible proxy.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", default=PORT, type=int)
    args = parser.parse_args()
    config = uvicorn.Config(
        "proxy.atlas_proxy:app",
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
