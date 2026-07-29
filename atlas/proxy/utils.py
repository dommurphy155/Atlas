"""JSON helpers, request IDs, and miscellaneous utilities."""

from __future__ import annotations

import uuid
from typing import Any, Union

import orjson
from fastapi import Request


def dumps(obj: Any) -> bytes:
    return orjson.dumps(obj)


def loads(data: Union[bytes, str]) -> Any:
    if isinstance(data, str):
        data = data.encode()
    return orjson.loads(data)


def request_id(request: Request) -> str:
    return (
        request.headers.get("x-request-id")
        or request.headers.get("x-client-request-id")
        or uuid.uuid4().hex[:16]
    )


def is_openai_done_frame(frame: bytes) -> bool:
    """True for OpenAI-style end markers that break some Anthropic clients."""
    text = frame.replace(b"\r\n", b"\n").strip()
    if not text:
        return True
    for line in text.split(b"\n"):
        line = line.strip()
        if line.startswith(b"data:"):
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return True
        if line == b"event: data":
            if b"[DONE]" in text:
                return True
    return False
