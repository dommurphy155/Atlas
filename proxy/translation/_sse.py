"""OpenAI chat/completions SSE -> Anthropic /messages SSE conversion.

The streaming translator is stateful across calls: it allocates Anthropic
content-block indices as new block types (thinking, text, tool_use) appear
in the upstream deltas, and emits the corresponding ``content_block_start``
/ ``content_block_stop`` pairs around their deltas.

The caller owns the ``state`` dict and passes it to every call so a
multi-chunk stream stays consistent. State keys:

  text_open        (bool)  : whether the text block is currently open
  text_index       (int)   : Anthropic index for the text block
  thinking_open    (bool)  : whether the thinking block is currently open
  thinking_index   (int)   : Anthropic index for the thinking block
  next_index       (int)   : next free Anthropic block index
  tool_blocks      (dict)  : per-OpenAI-tool-index -> {anthropic_idx, name,
                             started}

The two nested helpers ``_ensure_thinking`` and ``_ensure_text`` close
over the local ``out`` and ``state`` so they must stay as inner functions
(not module-level) to keep their closure semantics.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ._responses import _map_openai_finish_reason_to_anthropic


def openai_sse_to_anthropic_sse(
    sse_chunk: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Optional[Dict]]]:
    """Convert a single OpenAI SSE payload into Anthropic SSE events."""
    out: List[Tuple[str, Optional[Dict]]] = []

    if state is None:
        state = {}
    state.setdefault("text_open", False)
    state.setdefault("text_index", -1)
    state.setdefault("thinking_open", False)
    state.setdefault("thinking_index", -1)
    state.setdefault("next_index", 0)
    state.setdefault("tool_blocks", {})

    def _ensure_thinking() -> int:
        if not state["thinking_open"]:
            state["thinking_index"] = state["next_index"]
            state["next_index"] += 1
            state["thinking_open"] = True
            out.append(("content_block_start", {
                "type": "content_block_start",
                "index": state["thinking_index"],
                "content_block": {"type": "thinking", "thinking": ""},
            }))
        return state["thinking_index"]

    def _ensure_text() -> int:
        if not state["text_open"]:
            state["text_index"] = state["next_index"]
            state["next_index"] += 1
            state["text_open"] = True
            out.append(("content_block_start", {
                "type": "content_block_start",
                "index": state["text_index"],
                "content_block": {"type": "text", "text": ""},
            }))
        return state["text_index"]

    choices = sse_chunk.get("choices") or []
    if not choices:
        return out

    choice = choices[0]
    delta = choice.get("delta") or {}
    finish_reason = choice.get("finish_reason")

    # --- reasoning / thinking (must precede text per Anthropic spec) ---
    reasoning = delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        idx = _ensure_thinking()
        out.append(("content_block_delta", {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "thinking_delta", "thinking": reasoning},
        }))

    # --- text content ---
    content = delta.get("content")
    if isinstance(content, str) and content:
        idx = _ensure_text()
        out.append(("content_block_delta", {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "text_delta", "text": content},
        }))

    # --- tool calls ---
    tcs = delta.get("tool_calls") or []

    for tc in tcs:
        if not isinstance(tc, dict):
            continue

        idx = tc.get("index", 0)
        fn = tc.get("function") or {}

        if idx not in state["tool_blocks"]:
            anthropic_idx = state["next_index"]
            state["next_index"] += 1
            state["tool_blocks"][idx] = {
                "anthropic_idx": anthropic_idx,
                "name": "",
                "started": False,
            }

        block = state["tool_blocks"][idx]

        if fn.get("name") and not block["started"]:
            block["started"] = True
            block["name"] = fn["name"]
            out.append(("content_block_start", {
                "type": "content_block_start",
                "index": block["anthropic_idx"],
                "content_block": {
                    "type": "tool_use",
                    "id": tc.get("id") or "",
                    "name": fn["name"],
                    "input": {},
                },
            }))

        arguments = fn.get("arguments")
        if arguments:
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)

            out.append(("content_block_delta", {
                "type": "content_block_delta",
                "index": block["anthropic_idx"],
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": arguments,
                },
            }))

    # --- finish ---
    if finish_reason:
        # Close any still-open blocks before the message_stop.
        if state["thinking_open"]:
            out.append(("content_block_stop", {
                "type": "content_block_stop",
                "index": state["thinking_index"],
            }))
            state["thinking_open"] = False
        if state["text_open"]:
            out.append(("content_block_stop", {
                "type": "content_block_stop",
                "index": state["text_index"],
            }))
            state["text_open"] = False
        for idx, block in state["tool_blocks"].items():
            if block["started"]:
                out.append(("content_block_stop", {
                    "type": "content_block_stop",
                    "index": block["anthropic_idx"],
                }))
                block["started"] = False

        out.append(("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason":
                    _map_openai_finish_reason_to_anthropic(finish_reason),
            },
        }))

        out.append(("message_stop", {
            "type": "message_stop",
        }))
    return out
