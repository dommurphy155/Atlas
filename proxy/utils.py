"""JSON helpers, request IDs, and miscellaneous utilities.

Note: SSE frame classification and DONE detection live in
`proxy.streaming_sse` (extracted in Round 3). They were previously here
but moved to break the proxy.py / utils.py cross-module coupling that
the architecture review flagged.
"""

from __future__ import annotations

import uuid
from typing import Any, Union

import orjson
from fastapi import Request, WebSocket


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


def ws_request_id(ws: WebSocket) -> str:
    return (
        ws.headers.get("x-request-id")
        or ws.headers.get("x-client-request-id")
        or uuid.uuid4().hex[:16]
    )
