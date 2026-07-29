"""FastAPI route handlers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from .config import (
    FORCE_DEFAULT_MODEL,
    LISTEN_HOST,
    LISTEN_PORT,
    OPENROUTER_CHAT,
    OPENROUTER_MESSAGES,
    OPENROUTER_MODEL,
    OPENROUTER_MODELS,
    get_logger,
)
from .proxy import ProxyCore
from .translation import prepare_chat_body, prepare_messages_body
from .utils import dumps, loads, request_id

log = get_logger(__name__)

router = APIRouter()

# Set by main.py during lifespan startup
proxy: Optional[ProxyCore] = None


@router.get("/")
async def root() -> Dict[str, Any]:
    assert proxy is not None
    return {
        "service": "OpenRouter Translation Proxy",
        "version": "1.1.0",
        "default_model": OPENROUTER_MODEL,
        "force_default_model": FORCE_DEFAULT_MODEL,
        "endpoints": [
            "POST /v1/chat/completions",
            "POST /v1/responses",
            "POST /v1/messages",
            "GET  /v1/models",
            "GET  /health",
            "GET  /health/keys",
        ],
        "keys_loaded": proxy.pool.stats()["total"],
    }


@router.get("/health")
async def health() -> Dict[str, Any]:
    assert proxy is not None
    return {
        "status": "ok",
        "keys": proxy.pool.stats(),
        "listen": f"{LISTEN_HOST}:{LISTEN_PORT}",
    }


@router.get("/health/keys")
async def health_keys() -> Dict[str, Any]:
    """Detailed per-key statistics (no secret material)."""
    assert proxy is not None
    return {
        "status": "ok",
        "summary": proxy.pool.stats(),
        "keys": proxy.pool.detailed_stats(),
    }


@router.get("/stats")
async def stats() -> Dict[str, Any]:
    """Legacy /stats endpoint for atlas CLI compatibility."""
    assert proxy is not None
    stats = proxy.pool.stats()
    return {
        "total_keys": stats["total"],
        "healthy_keys": stats["healthy"],
        "cooling_keys": stats["cooling"],
        "suspended_keys": stats["suspended"],
    }


@router.get("/v1/models")
async def models(request: Request) -> Response:
    assert proxy is not None
    rid = request_id(request)
    return await proxy.forward("GET", OPENROUTER_MODELS, request_id=rid)


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    assert proxy is not None
    rid = request_id(request)
    try:
        body = loads(await request.body())
    except Exception:
        return JSONResponse(
            {
                "error": {
                    "message": "invalid json",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
            headers={"x-request-id": rid},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {
                "error": {
                    "message": "body must be object",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
            headers={"x-request-id": rid},
        )

    body = prepare_chat_body(body)
    stream = bool(body.get("stream", False))
    payload = dumps(body)

    log.info(
        "req=%s provider=openai endpoint=chat/completions model=%s stream=%s",
        rid,
        body.get("model"),
        stream,
    )
    return await proxy.forward(
        "POST",
        OPENROUTER_CHAT,
        body=payload,
        stream=stream,
        request_id=rid,
    )


@router.post("/v1/messages")
@router.post("/messages")
async def messages(request: Request) -> Response:
    assert proxy is not None
    rid = request_id(request)
    try:
        body = loads(await request.body())
    except Exception:
        return JSONResponse(
            {
                "error": {
                    "message": "invalid json",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
            headers={"x-request-id": rid},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {
                "error": {
                    "message": "body must be object",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
            headers={"x-request-id": rid},
        )

    body = prepare_messages_body(body)
    stream = bool(body.get("stream", False))
    payload = dumps(body)

    extra: Dict[str, str] = {}
    if "anthropic-version" in request.headers:
        extra["anthropic-version"] = request.headers["anthropic-version"]
    else:
        extra["anthropic-version"] = "2023-06-01"
    if "anthropic-beta" in request.headers:
        extra["anthropic-beta"] = request.headers["anthropic-beta"]

    log.info(
        "req=%s provider=anthropic endpoint=messages model=%s stream=%s",
        rid,
        body.get("model"),
        stream,
    )
    return await proxy.forward(
        "POST",
        OPENROUTER_MESSAGES,
        body=payload,
        extra_headers=extra,
        stream=stream,
        request_id=rid,
    )


@router.post("/v1/responses")
async def responses(request: Request) -> Response:
    """
    OpenAI Responses API → best-effort map to OpenRouter chat/completions.
    Translates `input` / `instructions` into messages when needed.
    """
    assert proxy is not None
    rid = request_id(request)
    try:
        body = loads(await request.body())
    except Exception:
        return JSONResponse(
            {
                "error": {
                    "message": "invalid json",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
            headers={"x-request-id": rid},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {
                "error": {
                    "message": "body must be object",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
            headers={"x-request-id": rid},
        )

    if "messages" not in body and "input" in body:
        inp = body.pop("input")
        messages_list: List[Dict] = []
        if isinstance(inp, str):
            messages_list = [{"role": "user", "content": inp}]
        elif isinstance(inp, list):
            for item in inp:
                if isinstance(item, str):
                    messages_list.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content") or item.get("text") or ""
                    messages_list.append({"role": role, "content": content})
        body["messages"] = messages_list

    if "instructions" in body and "messages" in body:
        instr = body.pop("instructions")
        body["messages"] = [
            {"role": "system", "content": instr}
        ] + body["messages"]

    body = prepare_chat_body(body)
    stream = bool(body.get("stream", False))
    payload = dumps(body)

    log.info(
        "req=%s provider=openai endpoint=responses→chat model=%s stream=%s",
        rid,
        body.get("model"),
        stream,
    )
    return await proxy.forward(
        "POST",
        OPENROUTER_CHAT,
        body=payload,
        stream=stream,
        request_id=rid,
    )
