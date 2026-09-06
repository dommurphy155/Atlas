"""FastAPI route handlers."""

from __future__ import annotations
import time
import uuid

from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import (
    get_force_default_model,
    LISTEN_HOST,
    LISTEN_PORT,
    MAX_RETRIES,
    MODEL_CONTEXT_WINDOW,
    PAYLOAD_DIR,
    PROVIDER,
    SAVE_PAYLOAD_FILES,
    get_chat_url,
    get_messages_url,
    get_default_model,
    get_logger,
)
from .proxy import ProxyCore
from .translation import prepare_chat_body, prepare_messages_body, openai_response_to_anthropic, openai_sse_to_anthropic_sse
from .utils import dumps, loads, request_id, ws_request_id
from . import prettylog as pl

log = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Generic custom/freeform tool adaptation
# ---------------------------------------------------------------------------

_CUSTOM_TOOL_TYPES = frozenset({"custom", "freeform"})


def _atlas_custom_tool_map(tools):
    """Return request-local metadata for Codex custom/freeform tools."""
    result = {}
    if not isinstance(tools, list):
        return result

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        name = tool.get("name")
        tool_type = tool.get("type")

        if name and tool_type in _CUSTOM_TOOL_TYPES:
            result[name] = {
                "type": tool_type,
                "name": name,
            }

        # Some Codex/OpenAI hosted-tool payloads use a bare type without name.
        if tool_type == "apply_patch" and not name:
            result["apply_patch"] = {
                "type": "custom",
                "name": "apply_patch",
            }

    return result


def _atlas_unwrap_custom_tool_args(raw):
    """Undo Atlas' internal JSON transport wrapper for custom/freeform tools."""
    if not isinstance(raw, str):
        return raw or ""

    try:
        parsed = loads(raw)
    except Exception:
        return raw

    if isinstance(parsed, dict) and "input" in parsed:
        value = parsed["input"]
        return value if isinstance(value, str) else str(value)

    return raw


def _atlas_custom_tool_kind(name, custom_tools):
    """Return original tool type, or None for an ordinary function tool."""
    meta = custom_tools.get(name)
    if isinstance(meta, dict):
        return meta.get("type")
    return None

# Set by main.py during lifespan startup
proxy: Optional[ProxyCore] = None


@router.get("/")
async def root() -> Dict[str, Any]:
    assert proxy is not None
    return {
        "service": "OpenRouter Translation Proxy",
        "version": "1.1.0",
        "provider": PROVIDER,
        "default_model": get_default_model(),
        "force_default_model": get_force_default_model(),
        "endpoints": [
            "POST /v1/chat/completions",
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
@router.get("/models")
async def models(request: Request) -> Response:
    """Return Anthropic-shaped model IDs so Claude Code UI accepts them.
    Actual inference always uses the provider's default model via get_force_default_model().
    """
    rid = request_id(request)
    now = int(time.time())
    default_model = get_default_model()
    data = {
        "object": "list",
        "data": [
            {
                "id": "claude-opus-5",
                "object": "model",
                "created": now,
                "owned_by": "anthropic",
                "display_name": "Opus 5",
                "type": "model",
                "context_window": MODEL_CONTEXT_WINDOW,
            },
            {
                "id": "claude-opus-5[1m]",
                "object": "model",
                "created": now,
                "owned_by": "anthropic",
                "display_name": "Opus 5 (1M context)",
                "type": "model",
                "context_window": MODEL_CONTEXT_WINDOW,
            },
            {
                "id": "claude-sonnet-5",
                "object": "model",
                "created": now,
                "owned_by": "anthropic",
                "display_name": "Sonnet 5",
                "type": "model",
                "context_window": MODEL_CONTEXT_WINDOW,
            },
            {
                "id": "claude-sonnet-5[1m]",
                "object": "model",
                "created": now,
                "owned_by": "anthropic",
                "display_name": "Sonnet 5 (1M context)",
                "type": "model",
                "context_window": MODEL_CONTEXT_WINDOW,
            },
            {
                "id": "claude-haiku-4-5",
                "object": "model",
                "created": now,
                "owned_by": "anthropic",
                "display_name": "Haiku 4.5",
                "type": "model",
                "context_window": MODEL_CONTEXT_WINDOW,
            },
            {
                "id": default_model,
                "object": "model",
                "created": now,
                "owned_by": "openrouter" if PROVIDER != "huggingface" else "huggingface",
                "display_name": default_model,
                "type": "model",
                "context_window": MODEL_CONTEXT_WINDOW,
            },
        ],
    }
    return JSONResponse(data, headers={"x-request-id": rid})



@router.post("/v1/responses")
async def responses_api(request: Request) -> Response:
    """OpenAI Responses API endpoint - translates chat/completions SSE into
    proper Responses API SSE events (response.created / output_item /
    content_part / output_text.delta / .done / response.completed)."""
    assert proxy is not None
    rid = request_id(request)
    try:
        body = loads(await request.body())
    except Exception:
        return JSONResponse(
            {"error": {"message": "invalid json", "type": "invalid_request_error"}},
            status_code=400,
            headers={"x-request-id": rid},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": {"message": "body must be object", "type": "invalid_request_error"}},
            status_code=400,
            headers={"x-request-id": rid},
        )


    # Convert Responses API shape (input/instructions) to chat messages
    if "input" in body or "instructions" in body:
        msgs = []
        instructions = body.pop("instructions", None)
        if instructions:
            msgs.append({"role": "system", "content": instructions})
        for item in body.pop("input", []) or []:
            itype = item.get("type")

            if itype == "function_call":
                msgs.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments", ""),
                        },
                    }],
                })
                continue

            if itype == "function_call_output":
                msgs.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", ""),
                })
                continue

            # Codex uses custom_tool_call (freeform tools, e.g. apply_patch).
            #
            # The synthesized tool schema in translation.py declares a single
            # `input` field, so arguments MUST be a JSON-encoded object like
            # {"input": "<patch text>"} when the round-tripped assistant
            # message reaches OpenRouter. Codex stores the raw patch in
            # item["input"] -- we wrap it in the schema-shaped object and
            # JSON-encode the whole thing so chat/completions parses it.
            #
            # If the stored input already looks like a JSON object (legacy
            # function-style custom tools), pass it through JSON-encoded as-is.
            if itype == "custom_tool_call":
                import json as _json
                raw_input = item.get("input", "")
                if isinstance(raw_input, str) and raw_input.strip().startswith("{"):
                    try:
                        parsed = _json.loads(raw_input)
                        if isinstance(parsed, dict):
                            args_str = raw_input
                        else:
                            args_str = _json.dumps({"input": raw_input})
                    except Exception:
                        args_str = _json.dumps({"input": raw_input})
                else:
                    args_str = _json.dumps({"input": raw_input})
                msgs.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": args_str,
                        },
                    }],
                })
                continue

            if itype == "custom_tool_call_output":
                msgs.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", ""),
                })
                continue

            # OpenAI hosted apply_patch tool wire format.
            # The diff payload lives at item["input"] on the Responses API.
            # Older payloads (and some forks) used item["operation"]["diff"];
            # fall back to that so we do not break previously-recorded
            # conversation history.
            #
            # arguments must be a JSON-encoded object {"input": "<diff>"} to
            # match the synthesized schema in translation.py. If the stored
            # payload already looks like a JSON object, pass it through as-is.
            if itype == "apply_patch_call":
                import json as _json
                raw = item.get("input")
                if raw is None:
                    raw = item.get("operation", {}).get("diff", "")
                if isinstance(raw, str) and raw.strip().startswith("{"):
                    args_str = raw  # already a JSON object string
                else:
                    args_str = _json.dumps({"input": raw})
                msgs.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": "apply_patch",
                            "arguments": args_str,
                        },
                    }],
                })
                continue

            if itype == "apply_patch_call_output":
                msgs.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", ""),
                })
                continue

            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            parts = item.get("content", [])
            text = "\n".join(
                p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
            )
            if text:
                msgs.append({"role": role, "content": text})
        if msgs:
            body["messages"] = msgs

    # Responses-only structured output (title gen etc) - drop, chat/completions
    # doesn't support text.format json_schema the same way
    body.pop("text", None)
    body.pop("store", None)
    body.pop("include", None)

    # Capture the original Codex tool kinds BEFORE prepare_chat_body()
    # converts custom/freeform tools into OpenAI function tools.
    original_custom_tools = _atlas_custom_tool_map(body.get("tools"))

    body = prepare_chat_body(body)
    stream = bool(body.get("stream", False))
    payload = dumps(body)
    _payload_path = _dump_payload(rid, "", body)

    tr = pl.trace(rid)
    tr.start("openai", body.get("model"), "responses", stream)

    resp = await proxy.forward(
        "POST",
        get_chat_url(),
        body=payload,
        stream=stream,
        request_id=rid,
    )

    if not stream or not hasattr(resp, "body_iterator"):
        return resp

    response_id = f"resp_{rid}"
    item_id = f"msg_{rid}"
    model_name = body.get("model")

    async def translate_stream():
        text_so_far = []
        started_item = False
        buf = b""
        tool_call_state = {}

        yield _sse_event("response.created", {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": model_name,
            },
        })

        async for raw in resp.body_iterator:
            if not raw:
                continue
            chunk = raw if isinstance(raw, (bytes, bytearray)) else raw.encode("utf-8")
            buf += chunk

            # Split on double-newline SSE frame boundaries; keep remainder buffered
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                data_line = None
                for line in frame.split(b"\n"):
                    line = line.strip()
                    if line.startswith(b"data:"):
                        data_line = line[5:].strip()
                        break
                if data_line is None:
                    continue
                if data_line == b"[DONE]":
                    continue
                try:
                    data = loads(data_line)
                except Exception:
                    continue

                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}

                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        call_id = tc.get("id")
                        fn = tc.get("function", {}) or {}
                        name = fn.get("name")
                        args_piece = fn.get("arguments", "")
                        custom_kind = _atlas_custom_tool_kind(name, original_custom_tools)

                        if idx not in tool_call_state:
                            tool_call_state[idx] = {
                                "id": call_id or f"call_{rid}_{idx}",
                                "name": name or "",
                                "args": "",
                                "custom_kind": custom_kind,
                            }

                            if custom_kind in _CUSTOM_TOOL_TYPES:
                                yield _sse_event("response.output_item.added", {
                                    "type": "response.output_item.added",
                                    "output_index": idx + 1,
                                    "item": {
                                        "id": f"ctc_{tool_call_state[idx]['id']}",
                                        "call_id": tool_call_state[idx]["id"],
                                        "type": "custom_tool_call",
                                        "status": "in_progress",
                                        "name": tool_call_state[idx]["name"],
                                        "input": "",
                                    },
                                })
                            else:
                                yield _sse_event("response.output_item.added", {
                                    "type": "response.output_item.added",
                                    "output_index": idx + 1,
                                    "item": {
                                        "id": f"fc_{tool_call_state[idx]['id']}",
                                        "call_id": tool_call_state[idx]["id"],
                                        "type": "function_call",
                                        "status": "in_progress",
                                        "name": tool_call_state[idx]["name"],
                                        "arguments": "",
                                    },
                                })

                        if name and not tool_call_state[idx]["name"]:
                            tool_call_state[idx]["name"] = name
                            tool_call_state[idx]["custom_kind"] = (
                                _atlas_custom_tool_kind(name, original_custom_tools)
                            )

                        if args_piece:
                            tool_call_state[idx]["args"] += args_piece

                            if tool_call_state[idx]["custom_kind"] not in _CUSTOM_TOOL_TYPES:
                                yield _sse_event("response.function_call_arguments.delta", {
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": tool_call_state[idx]["id"],
                                    "output_index": idx + 1,
                                    "delta": args_piece,
                                })

                content = delta.get("content")
                if content:
                    if not started_item:
                        yield _sse_event("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": item_id,
                                "type": "message",
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        })
                        yield _sse_event("response.content_part.added", {
                            "type": "response.content_part.added",
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""},
                        })
                        started_item = True

                    text_so_far.append(content)
                    yield _sse_event("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": content,
                    })

        full_text = "".join(text_so_far)

        for idx, tc in tool_call_state.items():
            if tc.get("custom_kind") in _CUSTOM_TOOL_TYPES:
                custom_input = _atlas_unwrap_custom_tool_args(tc["args"])

                yield _sse_event("response.custom_tool_call_arguments.done", {
                    "type": "response.custom_tool_call_arguments.done",
                    "item_id": tc["id"],
                    "output_index": idx + 1,
                    "input": custom_input,
                })

                yield _sse_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx + 1,
                    "item": {
                        "id": f"ctc_{tc['id']}",
                        "call_id": tc["id"],
                        "type": "custom_tool_call",
                        "status": "completed",
                        "name": tc["name"],
                        "input": custom_input,
                    },
                })
            else:
                yield _sse_event("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "item_id": tc["id"],
                    "output_index": idx + 1,
                    "arguments": tc["args"],
                })
                yield _sse_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx + 1,
                    "item": {
                        "id": f"fc_{tc['id']}",
                        "call_id": tc["id"],
                        "type": "function_call",
                        "status": "completed",
                        "name": tc["name"],
                        "arguments": tc["args"],
                    },
                })

        if started_item:
            yield _sse_event("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": full_text,
            })
            yield _sse_event("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": full_text},
            })
            yield _sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": full_text}],
                },
            })

        final_output = []
        for idx, tc in tool_call_state.items():
            if tc.get("custom_kind") in _CUSTOM_TOOL_TYPES:
                custom_input = _atlas_unwrap_custom_tool_args(tc["args"])

                final_output.append({
                    "id": f"ctc_{tc['id']}",
                    "call_id": tc["id"],
                    "type": "custom_tool_call",
                    "status": "completed",
                    "name": tc["name"],
                    "input": custom_input,
                })
            else:
                final_output.append({
                    "id": f"fc_{tc['id']}",
                    "call_id": tc["id"],
                    "type": "function_call",
                    "status": "completed",
                    "name": tc["name"],
                    "arguments": tc["args"],
                })
        if started_item:
            final_output.append({
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": full_text}],
            })

        yield _sse_event("response.completed", {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "completed",
                "model": model_name,
                "output": final_output,
            },
        })

    return StreamingResponse(translate_stream(), media_type="text/event-stream")



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

    _payload_path = _dump_payload(rid, "", body)

    tr = pl.trace(rid)  # logging-only lifecycle trace
    tr.start("openai", body.get("model"), "chat/completions", stream)

    resp = await proxy.forward(
        "POST",
        get_chat_url(),
        body=payload,
        stream=stream,
        request_id=rid,
    )
    # Non-streaming: promote reasoning_content → content for OpenAI clients
    # when the model emitted content: null + reasoning_content: <text>.
    # See translation.promote_reasoning_to_content_in_chat_response.
    if not stream and isinstance(resp, Response):
        try:
            data = loads(resp.body)
            if isinstance(data, dict):
                from .translation import promote_reasoning_to_content_in_chat_response
                promote_reasoning_to_content_in_chat_response(data)
                new_body = dumps(data)
                # Only swap if the body actually changed to avoid an
                # unnecessary re-encode on every response.
                if new_body != resp.body:
                    return Response(
                        content=new_body,
                        status_code=resp.status_code,
                        headers={k: v for k, v in resp.headers.items()
                                if k.lower() != "content-length"},
                        media_type=resp.headers.get("content-type", "application/json"),
                    )
        except Exception:
            # If promotion fails for any reason, return the original body.
            pass
    return resp

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

    body = prepare_messages_body(body) if PROVIDER != "huggingface" else prepare_chat_body(body)
    stream = bool(body.get("stream", False))
    payload = dumps(body)

    _payload_path = _dump_payload(rid, "_messages", body)

    tr = pl.trace(rid)  # logging-only lifecycle trace
    tr.start("anthropic", body.get("model"), "messages", stream)

    extra: Dict[str, str] = {}
    if "anthropic-version" in request.headers:
        extra["anthropic-version"] = request.headers["anthropic-version"]
    else:
        extra["anthropic-version"] = "2023-06-01"
    if "anthropic-beta" in request.headers:
        extra["anthropic-beta"] = request.headers["anthropic-beta"]

    upstream_url = get_chat_url() if PROVIDER == "huggingface" else get_messages_url()

    # HF streaming: open the upstream via iter_upstream_sse so mid-stream
    # provider errors classify + mark the key. The non-streaming path
    # still goes through proxy.forward (one-shot, no in-flight concerns).
    if PROVIDER == "huggingface" and stream:
        return await _stream_openai_to_anthropic(
            rid, "POST", upstream_url, payload, extra
        )

    resp = await proxy.forward(
        "POST",
        upstream_url,
        body=payload,
        extra_headers=extra,
        stream=stream,
        request_id=rid,
    )

    # HF non-streaming: convert OpenAI response → Anthropic /messages shape
    if PROVIDER == "huggingface" and isinstance(resp, Response):
        openai_data = loads(resp.body)
        anthropic_data = openai_response_to_anthropic(openai_data, rid=rid)
        new_body = dumps(anthropic_data)
        return Response(
            content=new_body,
            status_code=resp.status_code,
            media_type="application/json",
            headers={k: v for k, v in resp.headers.items() if k.lower() != "content-length"},
        )

    return resp



async def _stream_openai_to_anthropic(
    rid: str,
    method: str,
    upstream_url: str,
    payload: bytes,
    extra_headers: Dict[str, str],
) -> StreamingResponse | Response:
    """Stream an OpenAI SSE response from HF and re-emit as Anthropic SSE.

    Opens the upstream via ProxyCore.iter_upstream_sse so that:
    - mid-stream provider errors (rate_limit, concurrency, generic_error)
      classify and mark the key (so the next request rotates off it)
    - SSE chunk-boundary buffering is correct (frames split across httpx
      chunks are re-assembled before translation)
    - keepalives + connection cleanup are handled by the shared path
    """
    from .proxy import proxy as _proxy_singleton  # late import: avoid circular
    key, key_idx, is_healthy = await _proxy_singleton.pool.next_key_locked()
    if not is_healthy:
        s = _proxy_singleton.pool.stats()
        log.warning(
            "req=%s all keys unhealthy (healthy=%d cooling=%d suspended=%d) — fast-failing with 503",
            rid, s["healthy"], s["cooling"], s["suspended"],
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
            headers={"x-request-id": rid},
        )
    key_str = _proxy_singleton.pool.get_key_string(key_idx) or ""
    headers = _proxy_singleton._headers(key_str, extra_headers)
    iter_or_response = await _proxy_singleton.iter_upstream_sse(
        method, upstream_url, headers, payload, key_idx, rid,
    )

    # iter_upstream_sse returns a Response on upstream failure (caller
    # returns it directly) and an AsyncIterator on success.
    if isinstance(iter_or_response, Response):
        return iter_or_response
    frame_iter = iter_or_response

    async def translate_stream() -> AsyncIterator[bytes]:
        # Yield the message_start scaffolding.  Block lifecycles (text /
        # thinking / tool_use) are managed inside openai_sse_to_anthropic_sse
        # so reasoning_content and text deltas get distinct Anthropic content
        # block indices (mixing them at index 0 is invalid per the Anthropic
        # SDK).
        yield _sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": rid,
                "type": "message",
                "role": "assistant",
                "model": get_default_model(),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

        # state tracks which Anthropic content blocks have been opened so
        # reasoning_content / text / tool_use each get a distinct index.
        state: Dict[str, Any] = {
            "text_open": False,
            "text_index": -1,
            "thinking_open": False,
            "thinking_index": -1,
            "next_index": 0,
            "tool_blocks": {},
        }

        # iter_upstream_sse yields raw SSE frames (after chunk-buffering and
        # mid-stream error classification). [DONE] is filtered upstream on
        # the Anthropic translation path.
        async for frame in frame_iter:
            # frame is raw bytes like b"data: {...}\n\n" — extract payload
            data_line = None
            for line in frame.split(b"\n"):
                if line.startswith(b"data:"):
                    data_line = line[5:].strip()
                    break
            if not data_line or data_line == b"[DONE]":
                continue
            try:
                chunk = loads(data_line)
            except Exception:
                continue

            # HF SSE per-chunk diagnostics (debug only — too noisy for INFO)
            try:
                log.debug(
                    "req=%s HF_SSE_CHUNK %s",
                    rid,
                    data_line.decode("utf-8", errors="replace")[:4000],
                )
            except Exception:
                pass

            # Emit Anthropic events for each OpenAI delta. The state dict
            # carries block-lifecycle across chunks so thinking and text
            # get distinct content_block indices.
            for evt_name, evt_data in openai_sse_to_anthropic_sse(chunk, state):
                if evt_data is None:
                    continue
                yield _sse_event(evt_name, evt_data)

        # Final message_stop. Per-chunk blocks are closed inside the
        # translator when finish_reason arrives; emit message_stop for
        # streams that ended without a finish_reason (rare but possible).
        yield _sse_event("message_stop", {
            "type": "message_stop",
        })

    return StreamingResponse(
        translate_stream(),
        media_type="text/event-stream",
        headers={
            "x-request-id": rid,
            "cache-control": "no-cache, no-transform",
            "x-accel-buffering": "no",
        },
    )


def _sse_event(event: str, data: dict) -> bytes:
    raw = dumps(data)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return f"event: {event}\ndata: {raw}\n\n".encode("utf-8")


def _dump_payload(rid: str, tag: str, body: Any) -> str:
    """Write the request payload to PAYLOAD_DIR when SAVE_PAYLOAD_FILES is on.

    Returns the path written ("" when disabled or on failure). Never raises.
    """
    if not SAVE_PAYLOAD_FILES:
        return ""
    import os
    import orjson

    os.makedirs(PAYLOAD_DIR, exist_ok=True)
    path = os.path.join(PAYLOAD_DIR, f"payload_{rid}{tag}.json")
    try:
        with open(path, "w") as f:
            f.write(orjson.dumps(body, option=orjson.OPT_INDENT_2).decode())
        log.info("req=%s payload_saved %s (%d bytes)", rid, path, os.path.getsize(path))
    except Exception:
        return ""
    return path


