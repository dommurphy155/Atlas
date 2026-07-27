#!/usr/bin/env python3
"""Baseline benchmark — measures proxy overhead, not upstream latency.

Captures:
- Cold import time (time to import atlas_proxy and all deps)
- Module load (time for app object to be ready)
- Conversion pipeline throughput (anthropic_openai_payload + sanitize + responses_request_to_openai)
- SSE conversion throughput (openai_sse_to_anthropic_sse with mock chunks)

Run: python tests/benchmark.py
Saves results to tests/benchmark_baseline.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS: dict[str, object] = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "branch": os.popen("git branch --show-current").read().strip(),
    "commit": os.popen("git rev-parse --short HEAD").read().strip(),
}

# ── 1. Cold import time ──────────────────────────────────────────────

t0 = time.perf_counter()
from proxy.atlas_proxy import app  # noqa: E402
t1 = time.perf_counter()
RESULTS["cold_import_ms"] = round((t1 - t0) * 1000, 2)

# ── 2. Module-level objects ready ────────────────────────────────────

from proxy.openai_compat import (  # noqa: E402
    anthropic_openai_payload,
    sanitize_openai_payload,
    responses_request_to_openai,
    openai_sse_to_anthropic_sse,
    ResponsesRequest,
    ResponsesInput,
    ResponsesTool,
)

# ── 3. Conversion pipeline throughput ────────────────────────────────

# Realistic Anthropic request body
anthropic_body: dict[str, object] = {
    "model": "GLM-4.6",
    "max_tokens": 4096,
    "system": "You are a helpful assistant.",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
    ],
    "stream": False,
}

ITERATIONS = 1000

t0 = time.perf_counter()
for _ in range(ITERATIONS):
    payload = anthropic_openai_payload(anthropic_body, "GLM-4.6")
    sanitize_openai_payload(payload)
t1 = time.perf_counter()
RESULTS["anthropic_conversion_per_req_us"] = round((t1 - t0) / ITERATIONS * 1e6, 1)
RESULTS["anthropic_conversion_total_ms"] = round((t1 - t0) * 1000, 2)

# ── 4. Responses API conversion throughput ───────────────────────────

req = ResponsesRequest(
    model="GLM-4.6",
    input=ResponsesInput("What is the capital of France?"),
    instructions="You are a helpful assistant.",
    tools=[],
    tool_choice=None,
    stream=True,
)

t0 = time.perf_counter()
for _ in range(ITERATIONS):
    responses_request_to_openai(req)
t1 = time.perf_counter()
RESULTS["responses_conversion_per_req_us"] = round((t1 - t0) / ITERATIONS * 1e6, 1)
RESULTS["responses_conversion_total_ms"] = round((t1 - t0) * 1000, 2)

# ── 5. SSE conversion throughput ─────────────────────────────────────

import asyncio


async def _bench_sse() -> float:
    # Mock OpenAI SSE chunks
    chunks: list[bytes] = []
    for i in range(50):
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "GLM-4.6",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"word{i} "},
                    "finish_reason": None,
                }
            ],
        }
        chunks.append(f"data: {json.dumps(chunk)}\n\n".encode())
    # Final chunk
    final = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "GLM-4.6",
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}
        ],
    }
    chunks.append(f"data: {json.dumps(final)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")

    async def mock_upstream():
        for c in chunks:
            yield c

    t0 = time.perf_counter()
    for _ in range(100):
        async for _ in openai_sse_to_anthropic_sse(
            mock_upstream(),
            model="GLM-4.6",
        ):
            pass
    t1 = time.perf_counter()
    return round((t1 - t0) / 100 * 1000, 2)


sse_ms = asyncio.run(_bench_sse())
RESULTS["sse_conversion_per_stream_ms"] = sse_ms

# ── 6. Memory snapshot ──────────────────────────────────────────────

import resource

RESULTS["peak_rss_mb"] = round(
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
)

# ── Save ─────────────────────────────────────────────────────────────

output_path = PROJECT_ROOT / "tests" / "benchmark_baseline.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(RESULTS, indent=2))

print("Baseline benchmark results:")
print(json.dumps(RESULTS, indent=2))
print(f"\nSaved to {output_path}")
