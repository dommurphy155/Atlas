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
