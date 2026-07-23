"""Streaming engine with keepalive."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Optional

from src.core.types import Response
from src.streaming.sse import SSEFormatter


class StreamingEngine:
    """Streaming engine with keepalive support."""

    def __init__(self, keepalive_interval: float = 15.0):
        self.keepalive_interval = keepalive_interval

    async def stream(
        self,
        iterator: AsyncIterator[Response],
        format_chunk,
    ) -> AsyncIterator[bytes]:
        """Stream responses with keepalive."""
        yield b": ping\n\n"  # Immediate ping

        pending = None
        while True:
            if pending is None:
                pending = asyncio.create_task(iterator.__anext__())

            done, _ = await asyncio.wait(
                {pending}, timeout=self.keepalive_interval, return_when=asyncio.FIRST_COMPLETED
            )

            if not done:
                yield SSEFormatter.format_keepalive()
                continue

            try:
                chunk = pending.result()
            except StopAsyncIteration:
                return

            pending = None
            yield format_chunk(chunk)
