"""openai_sse_to_anthropic_sse — real-time OpenAI SSE → Anthropic SSE.

The translator is the most complex stateful code in the proxy. We drive it
with a fake byte iterator (no network) and assert the Anthropic event
sequence it emits. This is the single highest-value regression test: a silent
break here corrupts every /v1/messages stream.
"""
from __future__ import annotations

import json

import pytest

from proxy.openai_compat import openai_sse_to_anthropic_sse


def _oai_chunk(content: str | None = None, *, finish_reason: str | None = None,
                tool_calls=None, usage=None, chunk_id="chatcmpl-x") -> bytes:
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    choice = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    payload = {"id": chunk_id, "object": "chat.completion.chunk", "choices": [choice]}
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _aiter(chunks: list[bytes]):
    for c in chunks:
        yield c


def _parse_events(raw_bytes: bytes) -> list[tuple[str, dict]]:
    """Split an Anthropic SSE byte blob into (event, data) pairs."""
    events = []
    for block in raw_bytes.decode().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event and data:
            events.append((event, data))
    return events


@pytest.mark.asyncio
async def test_text_stream_emits_full_sequence():
    """A plain text stream must produce: message_start → content_block_start
    → content_block_delta(text) → content_block_stop → message_delta → message_stop."""
    chunks = [
        _oai_chunk("Hello"),
        _oai_chunk(" world"),
        _oai_chunk(finish_reason="stop", usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}),
        b"data: [DONE]\n\n",
    ]
    captured = []
    async for blob in openai_sse_to_anthropic_sse(_aiter(chunks), "claude"):
        captured.append(blob)
    raw = b"".join(captured)
    events = _parse_events(raw)

    event_names = [e for e, _ in events]
    assert event_names[0] == "message_start"
    assert "content_block_start" in event_names
    assert "content_block_delta" in event_names
    assert "content_block_stop" in event_names
    assert "message_delta" in event_names
    assert event_names[-1] == "message_stop"

    # Text deltas concatenated equal the full text.
    text = "".join(d["delta"]["text"] for e, d in events
                   if e == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert text == "Hello world"

    # stop_reason flows through from finish_reason=stop → end_turn.
    msg_delta = [d for e, d in events if e == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_tool_use_stream_emits_input_json_delta():
    """A tool-call stream must emit content_block_start(tool_use) +
    input_json_delta with the arguments fragment."""
    tc = [{"index": 0, "id": "call_1", "function": {"name": "search", "arguments": '{"q": "x"}'}}]
    chunks = [
        _oai_chunk(tool_calls=[{"index": 0, "id": "call_1", "function": {"name": "search", "arguments": '{"q":'}}]),
        _oai_chunk(tool_calls=[{"index": 0, "function": {"arguments": '"x"}'}}]),
        _oai_chunk(finish_reason="tool_calls"),
        b"data: [DONE]\n\n",
    ]
    captured = []
    async for blob in openai_sse_to_anthropic_sse(_aiter(chunks), "claude"):
        captured.append(blob)
    events = _parse_events(b"".join(captured))

    starts = [d for e, d in events if e == "content_block_start"]
    tool_starts = [d for d in starts if d["content_block"]["type"] == "tool_use"]
    assert tool_starts and tool_starts[0]["content_block"]["name"] == "search"

    json_deltas = [d for e, d in events if e == "content_block_delta"
                   and d["delta"]["type"] == "input_json_delta"]
    # The arguments fragments are forwarded as partial_json.
    joined = "".join(d["delta"]["partial_json"] for d in json_deltas)
    assert json.loads(joined) == {"q": "x"}

    msg_delta = [d for e, d in events if e == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_on_done_receives_usage():
    """The on_done callback must fire with the upstream usage tuple."""
    seen = {}
    def on_done(pt, ct, tt, tc):
        seen.update(pt=pt, ct=ct, tt=tt, tc=tc)

    chunks = [
        _oai_chunk("hi"),
        _oai_chunk(finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}),
        b"data: [DONE]\n\n",
    ]
    async for _ in openai_sse_to_anthropic_sse(_aiter(chunks), "claude", on_done=on_done):
        pass
    assert seen == {"pt": 10, "ct": 4, "tt": 14, "tc": 0}


@pytest.mark.asyncio
async def test_empty_stream_still_emits_valid_message():
    """An upstream that sends nothing must still yield a well-formed (empty)
    Anthropic stream, not a truncated connection."""
    captured = []
    async for blob in openai_sse_to_anthropic_sse(_aiter([]), "claude"):
        captured.append(blob)
    events = _parse_events(b"".join(captured))
    names = [e for e, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"


@pytest.mark.asyncio
async def test_upstream_error_chunk_surfaces_as_text_block():
    """A terminal upstream error chunk (e.g. mid-stream read timeout injected
    by NvidiaClient) must surface to the client as a text block, not vanish."""
    err = {"error": {"message": "upstream stream timed out (idle_read)", "rid": "abc123"}}
    chunks = [f"data: {json.dumps(err)}\n\n".encode(), b"data: [DONE]\n\n"]
    captured = []
    async for blob in openai_sse_to_anthropic_sse(_aiter(chunks), "claude"):
        captured.append(blob)
    text = b"".join(captured).decode()
    assert "stream error" in text
    assert "idle_read" in text
    assert "abc123" in text
