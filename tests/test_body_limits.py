"""Body-size guard tests."""
import pytest
from fastapi import Request
from starlette.requests import Request as StarletteRequest


@pytest.mark.asyncio
async def test_read_body_capped_under_limit():
    """Bodies under the cap should pass through unchanged."""
    from proxy.config import MAX_REQUEST_BODY_BYTES
    from proxy.routes import _read_body_capped

    # Build a real Starlette Request with a small body
    async def receive():
        return {"type": "http.request", "body": b'{"x":1}', "more_body": False}
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(b"content-length", str(len(b'{"x":1}')).encode())],
    }
    req = StarletteRequest(scope=scope, receive=receive)
    body = await _read_body_capped(req)
    assert body == b'{"x":1}'


@pytest.mark.asyncio
async def test_read_body_capped_over_limit_content_length():
    """Bodies over the cap with a Content-Length header should be rejected."""
    from proxy.config import MAX_REQUEST_BODY_BYTES
    from proxy.routes import _read_body_capped

    huge = b"x" * (MAX_REQUEST_BODY_BYTES + 1)
    async def receive():
        return {"type": "http.request", "body": huge, "more_body": False}
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(b"content-length", str(len(huge)).encode())],
    }
    req = StarletteRequest(scope=scope, receive=receive)
    with pytest.raises(ValueError, match="request body too large"):
        await _read_body_capped(req)


@pytest.mark.asyncio
async def test_read_body_capped_streaming_over_limit():
    """Bodies over the cap WITHOUT Content-Length (chunked upload) should
    be rejected while streaming (not after the whole body is buffered)."""
    from proxy.config import MAX_REQUEST_BODY_BYTES
    from proxy.routes import _read_body_capped

    # Simulate a chunked upload that's larger than the cap.
    chunk_count = 4
    chunk_size = MAX_REQUEST_BODY_BYTES // 2  # total = 2*cap → must overflow

    sent = [0]
    async def receive():
        if sent[0] >= chunk_count:
            return {"type": "http.disconnect"}
        sent[0] += 1
        return {
            "type": "http.request",
            "body": b"x" * chunk_size,
            "more_body": sent[0] < chunk_count,
        }

    # No content-length header
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
    }
    req = StarletteRequest(scope=scope, receive=receive)
    with pytest.raises(ValueError, match="request body too large"):
        await _read_body_capped(req)
