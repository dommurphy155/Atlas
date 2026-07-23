"""SSE streaming utilities."""

import json
from typing import Any, AsyncIterator, Optional


class SSEParser:
    """Parse SSE stream data."""

    @staticmethod
    def parse_line(line: bytes) -> Optional[dict[str, Any]]:
        line = line.strip()
        if not line.startswith(b"data:"):
            return None
        data = line[5:].strip()
        if data == b"[DONE]":
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None


class SSEFormatter:
    """Format data as SSE."""

    @staticmethod
    def format(data: dict[str, Any], event: Optional[str] = None) -> bytes:
        result = f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
        if event:
            result = f"event: {event}\n{result}"
        return result.encode()

    @staticmethod
    def format_text(text: str, model: str = "") -> bytes:
        chunk = {
            "choices": [{"delta": {"content": text}, "index": 0, "finish_reason": None}]
        }
        if model:
            chunk["model"] = model
        return SSEFormatter.format(chunk)

    @staticmethod
    def format_done() -> bytes:
        return b"data: [DONE]\n\n"

    @staticmethod
    def format_keepalive() -> bytes:
        return b": keepalive\n\n"


# Convenience function
def format_sse_event(data: str, event_type: Optional[str] = None) -> str:
    """Format SSE event as string for streaming response."""
    if not data and not event_type:
        return ": keepalive\n\n"
    if event_type == "ping":
        return f"event: ping\ndata: {data or ''}\n\n"
    if event_type == "done":
        return "data: [DONE]\n\n"
    if event_type == "message_delta":
        return f"data: {data}\n\n"
    if event_type == "message_stop":
        return f"event: message_stop\ndata: {data}\n\n"
    return f"data: {data}\n\n"
