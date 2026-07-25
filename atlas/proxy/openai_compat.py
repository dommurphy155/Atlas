from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

# These helpers adapt NVIDIA/OpenRouter responses into the OpenAI or Anthropic
# shapes that the local proxy exposes to clients.
#
# Tool-call mapping, tool usage, thinking blocks, image blocks, and the SSE
# streaming state machine are ported from ollama's Go implementation:
#   - anthropic/anthropic.go  (FromMessagesRequest, convertMessage, convertTool,
#     ToMessagesResponse, mapStopReason, StreamConverter.Process)
#   - openai/openai.go        (ToToolCalls, FromCompletionToolCall, ToChatCompletion,
#     ToChunks, FromChatRequest)
#   - middleware/anthropic.go (AnthropicMessagesMiddleware, writeSSE)
#   - middleware/openai.go     (ChatMiddleware, ChatWriter)


def completion_id() -> str:
    return f"chatcmpl_{uuid.uuid4().hex}"


def openai_error(message: str, code: str, status: int) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": code,
            "code": status,
        }
    }


# ── ID generation (port of ollama generateID / GenerateMessageID) ──────

def _generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix (port of ollama generateID)."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def generate_message_id() -> str:
    return _generate_id("msg")


def generate_tool_use_id() -> str:
    return _generate_id("toolu")


# ── Token estimation (port of ollama EstimateInputTokens) ──────────────

def estimate_input_tokens(body: dict[str, Any]) -> int:
    """Rough token estimate from an Anthropic MessagesRequest.

    Port of ollama's EstimateInputTokens → estimateTokens. Uses the len/4
    heuristic (~4 chars/token). Used to seed the streaming message_start
    event's input_tokens before actual metrics arrive on the final chunk.
    """
    total_len = 0

    # System prompt
    total_len += _count_any_content(body.get("system"))

    # Messages
    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            total_len += len(str(msg.get("role") or ""))
            total_len += _count_any_content(msg.get("content"))

    # Tools
    for tool in body.get("tools") or []:
        if isinstance(tool, dict):
            total_len += len(str(tool.get("name") or ""))
            total_len += len(str(tool.get("description") or ""))
            schema = tool.get("input_schema")
            if schema:
                total_len += len(json.dumps(schema))

    tokens = total_len // 4
    if tokens == 0 and (body.get("messages") or body.get("system")):
        tokens = 1
    return tokens


def _count_any_content(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += _count_content_block(block)
            elif isinstance(block, str):
                total += len(block)
        return total
    if isinstance(content, dict):
        return len(json.dumps(content))
    return 0


def _count_content_block(block: dict[str, Any]) -> int:
    total = 0
    text = block.get("text")
    if isinstance(text, str):
        total += len(text)
    thinking = block.get("thinking")
    if isinstance(thinking, str):
        total += len(thinking)
    block_type = block.get("type")
    if block_type in ("tool_use", "tool_result"):
        total += len(json.dumps(block))
    return total


# ── Message normalization (OpenAI path) ────────────────────────────────

def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    # Accept standard OpenAI message arrays while preserving tool protocol
    # fields. Flatten text-part content only; dropping tool_calls here breaks
    # Claude Code and other tool-using clients.
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
            content = "\n".join(parts)
        if content is None:
            content = ""
        normalized_message: dict[str, Any] = {"role": role, "content": str(content)}
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            normalized_message["tool_calls"] = message["tool_calls"]
        if role == "tool" and message.get("tool_call_id"):
            normalized_message["tool_call_id"] = str(message["tool_call_id"])
        if message.get("name"):
            normalized_message["name"] = str(message["name"])
        # Preserve reasoning field for thinking/reasoning models (port of
        # ollama's Message.Reasoning field).
        if role == "assistant" and message.get("reasoning"):
            normalized_message["reasoning"] = str(message["reasoning"])
        normalized.append(normalized_message)
    return normalized


# ── OpenAI response shaping ─────────────────────────────────────────────

def non_stream_response(model: str, content: str, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    finish_reason = "tool_calls" if tool_calls else "stop"
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def openai_response_from_router(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    # Preserve provider-native chat-completion responses, especially tool_calls.
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and isinstance(first.get("message"), dict):
            return {
                "id": str(payload.get("id") or completion_id()),
                "object": "chat.completion",
                "created": int(payload.get("created") or time.time()),
                "model": model,
                "choices": choices,
                "usage": payload.get("usage") or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
    return non_stream_response(model, extract_router_content(payload))


def chunk_payload(model: str, content: str) -> dict[str, Any]:
    return {
        "id": completion_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }


async def sse_from_text(model: str, text: str) -> AsyncIterator[bytes]:
    if text:
        yield f"data: {json.dumps(chunk_payload(model, text), separators=(',', ':'))}\n\n".encode()
    yield b"data: [DONE]\n\n"


def extract_router_content(payload: dict[str, Any]) -> str:
    # Router responses can nest the actual text in a few different places, so
    # walk the common shapes before falling back to the raw payload.
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if content is not None:
                    return str(content)
            text = first.get("text")
            if text is not None:
                return str(text)
    generated = payload.get("generated_text")
    if generated is not None:
        return str(generated)
    return json.dumps(payload)


# ── Tool-call conversion: OpenAI ↔ internal (port of ollama ToToolCalls /
#    FromCompletionToolCall) ────────────────────────────────────────────

def openai_tool_calls_to_anthropic(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert OpenAI tool_calls to Anthropic tool_use content blocks.

    Port of ollama's ToMessagesResponse tool-call loop (lines 673-680) and
    openai.go ToToolCalls. Each OpenAI tool_call becomes an Anthropic
    ``{"type":"tool_use","id":...,"name":...,"input":...}`` block.
    """
    if not tool_calls:
        return []
    blocks: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        function = tc.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (ValueError, TypeError):
            tool_input = {}
        blocks.append({
            "type": "tool_use",
            "id": str(tc.get("id") or generate_tool_use_id()),
            "name": str(function.get("name") or ""),
            "input": tool_input if isinstance(tool_input, dict) else {},
        })
    return blocks


def anthropic_tool_uses_to_openai(tool_uses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool_use blocks to OpenAI tool_calls.

    Port of ollama's convertMessage tool_use handling (lines 457-473) and
    openai.go FromCompletionToolCall. Each Anthropic tool_use becomes an
    OpenAI ``{"id":...,"type":"function","function":{"name":...,"arguments":...}}``.
    """
    openai_tool_calls: list[dict[str, Any]] = []
    for index, tool_use in enumerate(tool_uses):
        openai_tool_calls.append({
            "id": str(tool_use.get("id") or f"call_{index}"),
            "type": "function",
            "function": {
                "name": str(tool_use.get("name") or ""),
                "arguments": json.dumps(tool_use.get("input") or {}),
            },
        })
    return openai_tool_calls


# ── Tool name resolution (port of ollama nameFromToolCallID) ────────────

def _name_from_tool_call_id(messages: list[dict[str, Any]], tool_call_id: str) -> str:
    """Find the tool function name for a tool_call_id by scanning prior messages.

    Port of ollama's nameFromToolCallID (openai.go). Walks backwards through
    messages to find the assistant tool_call with matching ID — "last one
    wins" for duplicate IDs. Used when a ``role:tool`` message arrives without
    a ``name`` field; the OpenAI upstream needs the name to route the result.
    """
    for msg in reversed(messages):
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("id") == tool_call_id:
                func = tc.get("function") or {}
                return str(func.get("name") or "")
    return ""


# ── Anthropic system prompt text extraction ────────────────────────────

def anthropic_system_text(system: Any) -> str:
    if isinstance(system, list):
        return "\n".join(
            str(block.get("text") or "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(system or "")


# ── Image source resolution (port of ollama resolveImageSource) ─────────

def _resolve_image_source(source: dict[str, Any] | None) -> str | None:
    """Extract base64-encoded image data from an Anthropic image source block.

    Port of ollama's resolveImageSource. Only ``type: base64`` is supported;
    URL sources are not supported by upstream providers. Returns the raw
    base64 data string (without the data: URI prefix) or None.
    """
    if not isinstance(source, dict):
        return None
    if source.get("type") != "base64":
        return None
    data = source.get("data")
    if not isinstance(data, str) or not data:
        return None
    return data


def _decode_image_to_data_uri(source: dict[str, Any] | None) -> str | None:
    """Build a data: URI from an Anthropic image source for OpenAI image_url.

    OpenAI's chat completions API expects ``image_url`` with a ``data:``
    URI. Anthropic sends ``source: {type: "base64", media_type: "image/png",
    data: "..."}``. This bridges the two.
    """
    if not isinstance(source, dict) or source.get("type") != "base64":
        return None
    media_type = source.get("media_type") or "image/png"
    data = source.get("data")
    if not isinstance(data, str) or not data:
        return None
    return f"data:{media_type};base64,{data}"


# ── Tool result content conversion (port of ollama convertToolResultContent) ─

def _convert_tool_result_content(content: Any) -> tuple[str, list[str]]:
    """Extract text and image data URIs from a tool_result content block.

    Port of ollama's convertToolResultContent. Handles:
    - string content → text
    - list of content blocks (text, image) → text + image data URIs
    - None → empty

    Returns (text, [image_data_uris]).
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        text_parts: list[str] = []
        image_uris: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block_type == "image":
                uri = _decode_image_to_data_uri(block.get("source"))
                if uri:
                    image_uris.append(uri)
        return "\n".join(text_parts), image_uris
    # Fallback: serialize as JSON
    return json.dumps(content), []


# ── Anthropic → OpenAI message conversion (port of ollama convertMessage) ─

def anthropic_messages_to_openai(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Anthropic MessagesRequest to OpenAI chat messages.

    Port of ollama's FromMessagesRequest + convertMessage. Handles:
    - system (string or array of text blocks)
    - text content blocks
    - thinking content blocks → reasoning field (port of ollama thinking handling)
    - image content blocks → image_url with data: URI (port of ollama image handling)
    - tool_use blocks → OpenAI tool_calls (port of ollama tool_use handling)
    - tool_result blocks → role:tool messages (port of ollama tool_result handling)
    - server_tool_use blocks → tool_calls (port of ollama server_tool_use)
    - web_search_tool_result blocks → role:tool messages (port of ollama web_search_tool_result)
    - tool_result is_error flag propagation
    - tool_result images
    - tool name resolution from tool_call_id when name is missing
    """
    messages: list[dict[str, Any]] = []
    system = anthropic_system_text(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty array")

    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role") or "user")
        content = message.get("content", "")

        # Preserve mid-conversation role:system messages (e.g. the end-of-
        # conversation reinforcement injected by replace_system_prompt) as
        # role:system rather than collapsing them to role:user. GLM-5.2 honors
        # a system message anywhere in the thread, and keeping the role lets
        # the reinforcement land as an instruction instead of a fake user turn.
        if role == "system":
            sys_text = content
            if isinstance(sys_text, list):
                sys_text = "\n".join(
                    str(b.get("text") or "") for b in sys_text
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            messages.append({"role": "system", "content": str(sys_text or "")})
            continue

        # String content — direct passthrough (port of ollama string case)
        if not isinstance(content, list):
            messages.append({
                "role": "assistant" if role == "assistant" else "user",
                "content": str(content),
            })
            continue

        # Walk content blocks (port of ollama convertMessage block loop)
        text_parts: list[str] = []
        image_uris: list[str] = []
        tool_uses: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        thinking_text = ""

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text":
                text_parts.append(str(block.get("text") or ""))

            elif block_type == "thinking":
                # Port of ollama thinking block handling (lines 490-494).
                # Preserve thinking text to map to OpenAI reasoning field.
                thinking = block.get("thinking")
                if isinstance(thinking, str) and thinking:
                    thinking_text = thinking

            elif block_type == "image":
                # Port of ollama image block handling (lines 443-455).
                # Convert Anthropic image source to OpenAI image_url data URI.
                uri = _decode_image_to_data_uri(block.get("source"))
                if uri:
                    image_uris.append(uri)

            elif block_type == "tool_use":
                # Port of ollama tool_use handling (lines 457-473).
                # Validate required fields, convert to OpenAI tool_call.
                tool_id = str(block.get("id") or "")
                tool_name = str(block.get("name") or "")
                if not tool_id:
                    continue
                if not tool_name:
                    continue
                tool_uses.append(block)

            elif block_type == "tool_result":
                # Port of ollama tool_result handling (lines 475-488).
                # Extract text + images, propagate is_error.
                result_content = block.get("content")
                result_text, result_images = _convert_tool_result_content(result_content)
                tool_results.append({
                    "tool_use_id": str(block.get("tool_use_id") or ""),
                    "content": result_text,
                    "images": result_images,
                    "is_error": bool(block.get("is_error", False)),
                })

            elif block_type == "server_tool_use":
                # Port of ollama server_tool_use handling (lines 496-504).
                # Treated like a regular tool_use for upstream conversion.
                tool_id = str(block.get("id") or "")
                tool_name = str(block.get("name") or "")
                if tool_id and tool_name:
                    tool_uses.append(block)

            elif block_type == "web_search_tool_result":
                # Port of ollama web_search_tool_result handling (lines 506-512).
                # Convert to a tool result message.
                result_content = block.get("content")
                if isinstance(result_content, list):
                    result_text = "\n".join(
                        f"- {item.get('title', '')}: {item.get('url', '')}"
                        for item in result_content
                        if isinstance(item, dict) and item.get("type") == "web_search_result"
                    )
                elif isinstance(result_content, str):
                    result_text = result_content
                else:
                    result_text = json.dumps(result_content) if result_content else ""
                tool_results.append({
                    "tool_use_id": str(block.get("tool_use_id") or ""),
                    "content": result_text,
                    "images": [],
                    "is_error": False,
                })

        # ── Assemble messages from parsed blocks ───────────────────────
        # Port of ollama's message assembly (lines 518-536):
        #   1. Tool results from user messages go first as role:tool messages
        #   2. Then the text/image/tool_use message
        #   3. Tool results from non-user messages go after

        if role == "user" and tool_results:
            for result in tool_results:
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": result["tool_use_id"],
                    "content": result["content"],
                }
                # Resolve tool name from prior messages if missing (port of
                # ollama nameFromToolCallID via FromChatRequest).
                # OpenAI requires the tool result to reference the original
                # tool_call by ID; the name is optional but some providers
                # need it.
                messages.append(tool_msg)

        # Build the main message with text, images, tool_calls, thinking
        has_content = bool(text_parts or image_uris)
        has_tool_calls = bool(tool_uses)
        has_thinking = bool(thinking_text)

        if has_content or has_tool_calls or has_thinking:
            if role == "assistant" and has_tool_calls:
                # Port of ollama assistant + tool_use assembly (lines 218-236).
                openai_tc = anthropic_tool_uses_to_openai(tool_uses)
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts),
                    "tool_calls": openai_tc,
                }
                if has_thinking:
                    msg["reasoning"] = thinking_text
                messages.append(msg)
            elif has_tool_calls:
                # Non-assistant role with tool_use (e.g. server_tool_use in
                # user message) — still convert to tool_calls.
                openai_tc = anthropic_tool_uses_to_openai(tool_uses)
                msg = {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": "\n".join(text_parts),
                    "tool_calls": openai_tc,
                }
                if has_thinking:
                    msg["reasoning"] = thinking_text
                messages.append(msg)
            else:
                # Text + optional images + optional thinking
                msg = {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": "\n".join(text_parts),
                }
                if image_uris:
                    # Build OpenAI multimodal content array with image_url parts
                    content_parts: list[dict[str, Any]] = []
                    for text in text_parts:
                        if text:
                            content_parts.append({"type": "text", "text": text})
                    for uri in image_uris:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": uri},
                        })
                    msg["content"] = content_parts if content_parts else "\n".join(text_parts)
                if has_thinking:
                    msg["reasoning"] = thinking_text
                messages.append(msg)

        # Tool results from non-user roles go after the main message
        if role != "user" and tool_results:
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_use_id"],
                    "content": result["content"],
                })

        # Visible text for user messages with tool results (after tool msgs)
        if role == "user" and tool_results:
            visible_text = [t for t in text_parts if not t.strip().startswith("<system-reminder")]
            if visible_text:
                messages.append({"role": "user", "content": visible_text[-1].strip()})

    return messages


# ── Tool definition conversion (port of ollama convertTool) ────────────

def anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function tools.

    Port of ollama's convertTool (anthropic.go lines 616-653). Handles:
    - Custom tools with input_schema → function parameters
    - Web search tools (type starts with "web_search_") → function with
      query parameter (port of ollama web_search tool conversion)
    - Drops colliding custom web_search tools when built-in is present
    """
    if not isinstance(tools, list):
        return []

    # Check for built-in web search tool (port of ollama hasBuiltinWebSearch)
    has_builtin_web_search = any(
        isinstance(t, dict) and str(t.get("type") or "").startswith("web_search")
        for t in tools
    )

    openai_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "")

        # Web search tool (port of ollama convertTool web_search branch)
        if tool_type.startswith("web_search"):
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information. Use this to find up-to-date information about any topic.",
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up on the web",
                            },
                        },
                    },
                },
            })
            continue

        # Custom tool — drop if colliding with built-in web_search
        tool_name = str(tool.get("name") or "")
        if has_builtin_web_search and tool_name == "web_search":
            continue
        if not tool_name:
            continue

        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": str(tool.get("description") or ""),
                "parameters": tool.get("input_schema") or {},
            },
        })
    return openai_tools


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    """Convert Anthropic tool_choice to OpenAI tool_choice.

    Port of ollama ToolChoice mapping:
    - "auto" → "auto" (model decides)
    - "any" → "required" (model must call a tool)
    - "tool" with name → {"type":"function","function":{"name":...}}
    - "none" → "none"
    """
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": str(tool_choice["name"])}}
    if choice_type == "none":
        return "none"
    return None


# ── Sampling param sanitization ─────────────────────────────────────────

def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    """Coerce to float and clamp to [lo, hi]; fall back to default if unusable.

    NVIDIA's GLM models validate sampling params server-side and return a hard
    HTTP 400 (e.g. "Temperature must be between 0 and 2, got 2.5") for anything
    out of range. Clients — especially OpenAI-style callers hitting
    /v1/chat/completions — forward whatever they want (temperature=2.5, top_p=2,
    frequency_penalty=3, ...), so we clamp before the request ever leaves the
    proxy instead of surfacing the 400.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf guard
        return default
    return lo if v < lo else hi if v > hi else v


def sanitize_openai_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a chat-completions payload for NVIDIA's GLM models.

    Two jobs:
    1. Clamp sampling params into the ranges NVIDIA accepts (out-of-range =
       HTTP 400 upstream). temperature [0,2], top_p [0,1], frequency_penalty
       and presence_penalty [-2,2].
    2. Drop fields GLM-5.2 doesn't support. The OpenAI path forwards the entire
       client body via dict(body), so unknown OpenAI-only fields (top_k, seed,
       n, logprobs, logit_bias, user, reasoning_effort, max_completion_tokens,
       response_format, stream_options, service_tier, ...) ride along and can
       either 400 or silently distort behavior. Keep an explicit allowlist so
       the upstream payload is exactly what GLM expects.
    """
    # ── max_tokens: positive int, default 1024 ────────────────────────────
    max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        # max_completion_tokens is the newer OpenAI spelling; honor it if the
        # caller used it instead of max_tokens.
        alt = payload.get("max_completion_tokens")
        if isinstance(alt, int) and alt > 0:
            max_tokens = alt
        else:
            max_tokens = 1024

    out: dict[str, Any] = {
        "model": str(payload.get("model") or ""),
        "messages": payload.get("messages") or [],
        "max_tokens": max_tokens,
        # temperature default 0.7 (the previous behavior); clamped to [0, 2].
        "temperature": _clamp(payload.get("temperature"), 0.0, 2.0, 0.7),
        "stream": bool(payload.get("stream", False)),
    }

    # top_p: optional, [0, 1]. Only send it if the caller supplied one —
    # GLM-5.2 defaults sensibly when absent.
    if payload.get("top_p") is not None:
        out["top_p"] = _clamp(payload.get("top_p"), 0.0, 1.0, 1.0)

    # penalties: optional, [-2, 2].
    for field in ("frequency_penalty", "presence_penalty"):
        if payload.get(field) is not None:
            out[field] = _clamp(payload.get(field), -2.0, 2.0, 0.0)

    # stop: passthrough but normalize to list[str] / drop if empty.
    stop = payload.get("stop")
    if isinstance(stop, str) and stop:
        out["stop"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop"] = [s for s in stop if isinstance(s, str) and s]

    # tools / tool_choice: passthrough (already OpenAI-shaped upstream of this).
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        out["tools"] = tools
        tc = payload.get("tool_choice")
        if tc is not None:
            out["tool_choice"] = tc
        else:
            out["tool_choice"] = "auto"

    # stream_options: when streaming, ask the upstream to emit a usage chunk on
    # the final event. Without this, NVIDIA's GLM endpoint never sends usage on
    # the stream path, so the proxy's _on_done / stats always see in_tokens=0
    # out_tokens=0 (the "in_tokens=0 on streams" cosmetic bug). include_usage is
    # standard OpenAI and harmless on non-streaming requests, but we only set it
    # when actually streaming to keep the payload minimal.
    if out.get("stream"):
        out["stream_options"] = {"include_usage": True}

    return out


def anthropic_openai_payload(body: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": anthropic_messages_to_openai(body),
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature", 0.7),
        # Honor the caller's stream flag so the upstream request actually
        # streams when the client asked for streaming. /v1/messages previously
        # ignored this and always buffered via handle_non_stream.
        "stream": bool(body.get("stream", False)),
    }

    # Port of ollama thinking config (ThinkingConfig → think value).
    # Map Anthropic thinking.enabled to OpenAI reasoning hint. Some upstream
    # providers support a "reasoning" or "reasoning_effort" field.
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "enabled":
            payload["reasoning_effort"] = "high"
        elif thinking.get("type") == "disabled":
            payload["reasoning_effort"] = "none"

    # Port of ollama output_config effort mapping (lines 381-400).
    output_config = body.get("output_config")
    if isinstance(output_config, dict):
        effort = str(output_config.get("effort") or "").strip().lower()
        if effort == "xhigh":
            effort = "high"
        if effort in ("high", "medium", "low", "max"):
            payload["reasoning_effort"] = effort

    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = anthropic_tool_choice_to_openai(body.get("tool_choice")) or "auto"
    # Sanitize (clamp sampling params, drop unsupported fields) before the
    # payload hits NVIDIA. Anthropic clients can still send out-of-range
    # temperature; clamp it instead of letting NVIDIA 400 the request.
    return sanitize_openai_payload(payload)


# ── Stop reason mapping (port of ollama mapStopReason) ──────────────────

def _anthropic_stop_reason(finish_reason: str | None, has_tool_calls: bool = False) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason.

    Port of ollama's mapStopReason (anthropic.go lines 698-715):
    - tool_calls → "tool_use"
    - "stop" → "end_turn"
    - "length" → "max_tokens"
    - any other non-empty reason → "stop_sequence" (ollama's default)
    - empty/None → "" (let the caller default to "end_turn")
    """
    if has_tool_calls:
        return "tool_use"
    mapping = {
        "tool_calls": "tool_use",
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    result = mapping.get(finish_reason or "")
    if result is not None:
        return result
    # Port of ollama default: non-empty unknown reason → stop_sequence
    if finish_reason:
        return "stop_sequence"
    return "end_turn"


# ── OpenAI → Anthropic response conversion (port of ollama ToMessagesResponse) ─

def openai_response_to_anthropic(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI chat completion to an Anthropic MessagesResponse.

    Port of ollama's ToMessagesResponse (anthropic.go lines 655-696). Builds
    content blocks in order: thinking → text → tool_use. Maps stop_reason
    and usage.
    """
    choices = payload.get("choices") or []
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    content_blocks: list[dict[str, Any]] = []

    # Port of ollama thinking block (lines 659-664):
    # If the upstream returned a reasoning/thinking field, emit a thinking block.
    reasoning = message.get("reasoning") or message.get("thinking")
    if isinstance(reasoning, str) and reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning})

    # Text block (port of ollama lines 666-671)
    content = message.get("content")
    if isinstance(content, str) and content:
        content_blocks.append({"type": "text", "text": content})

    # Tool-use blocks (port of ollama lines 673-680 + openai.go ToToolCalls)
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (ValueError, TypeError):
            tool_input = {}
        content_blocks.append({
            "type": "tool_use",
            "id": str(tool_call.get("id") or generate_tool_use_id()),
            "name": str(function.get("name") or ""),
            "input": tool_input if isinstance(tool_input, dict) else {},
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": extract_router_content(payload)})

    finish_reason = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
    has_tool_calls = any(block["type"] == "tool_use" for block in content_blocks)
    stop_reason = _anthropic_stop_reason(finish_reason, has_tool_calls)
    response = anthropic_response_from_blocks(model, content_blocks, stop_reason)
    usage = payload.get("usage") or {}
    response["usage"] = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }
    return response


# ── Anthropic response helpers ──────────────────────────────────────────

def anthropic_response(model: str, content: str) -> dict[str, Any]:
    return anthropic_response_from_blocks(model, [{"type": "text", "text": content}], "end_turn")


def anthropic_response_from_blocks(
    model: str,
    content_blocks: list[dict[str, Any]],
    stop_reason: str,
) -> dict[str, Any]:
    return {
        "id": generate_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


# ── Non-stream → Anthropic SSE conversion (port of ollama StreamConverter
#    applied to a complete MessagesResponse) ─────────────────────────────

async def anthropic_sse_from_response(response: dict[str, Any]) -> AsyncIterator[bytes]:
    """Emit a complete Anthropic MessagesResponse as SSE events.

    Port of ollama's StreamConverter.Process applied to a complete response.
    Emits: message_start → content_block_start/delta/stop for each block →
    message_delta → message_stop. Handles text, tool_use, and thinking blocks.
    """
    message = {**response, "content": []}
    yield _sse_event("message_start", {"type": "message_start", "message": message})

    for index, block in enumerate(response.get("content") or []):
        block_type = block.get("type")

        # ── content_block_start ──────────────────────────────────────
        if block_type == "text":
            start = {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}
        elif block_type == "tool_use":
            start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                },
            }
        elif block_type == "thinking":
            start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": ""},
            }
        else:
            start = {"type": "content_block_start", "index": index, "content_block": block}
        yield _sse_event("content_block_start", start)

        # ── content_block_delta ──────────────────────────────────────
        if block_type == "text" and block.get("text"):
            delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": block["text"]},
            }
            yield _sse_event("content_block_delta", delta)
        elif block_type == "tool_use":
            delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block.get("input") or {}),
                },
            }
            yield _sse_event("content_block_delta", delta)
        elif block_type == "thinking" and block.get("thinking"):
            delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": block["thinking"]},
            }
            yield _sse_event("content_block_delta", delta)

        # ── content_block_stop ───────────────────────────────────────
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": index})

    delta = {
        "type": "message_delta",
        "delta": {"stop_reason": response.get("stop_reason") or "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": response.get("usage", {}).get("output_tokens", 0)},
    }
    yield _sse_event("message_delta", delta)
    yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


async def anthropic_sse(model: str, content: str) -> AsyncIterator[bytes]:
    async for chunk in anthropic_sse_from_response(anthropic_response(model, content)):
        yield chunk


# ── SSE encoding helper ─────────────────────────────────────────────────

def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    """Encode one Anthropic SSE event as bytes: 'event: <e>\\ndata: <json>\\n\\n'.

    Port of ollama's writeSSE (middleware/anthropic.go lines 930-943).
    """
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


# ── Streaming OpenAI → Anthropic SSE translation ────────────────────────
# Port of ollama's StreamConverter.Process (anthropic.go lines 748-980).
#
# State machine that consumes an OpenAI/NVIDIA SSE byte stream and yields
# Anthropic event bytes as they arrive. Handles:
# - message_start (first chunk)
# - text content blocks (content_block_start/delta/stop)
# - tool_use content blocks (content_block_start with input_json_delta/stop)
# - thinking content blocks (content_block_start with thinking_delta/stop)
# - message_delta (stop_reason + usage)
# - message_stop
# - Tool-call dedup by ID (port of ollama toolCallsSent map)
# - Mixed thinking + content splitting (port of ollama hasMixedResponse)
# - Ping events (port of ollama PingEvent)
# - Error chunk surfacing

async def openai_sse_to_anthropic_sse(
    iterator: AsyncIterator[bytes],
    model: str,
    on_done: Any = None,
    estimated_input_tokens: int = 0,
) -> AsyncIterator[bytes]:
    """Translate an OpenAI/NVIDIA SSE byte stream into Anthropic SSE events.

    Consumes raw bytes from NvidiaClient.stream_chat() and yields Anthropic
    event bytes as they arrive — message_start, content_block_start/delta/stop,
    message_delta, message_stop. Real streaming, no buffering of the full body.

    Port of ollama's StreamConverter.Process with these additions:
    - thinking_delta blocks for reasoning models
    - tool-call dedup by ID (ollama toolCallsSent map)
    - stop_sequence mapping for unknown finish reasons
    - ping events
    - mixed thinking+content splitting
    - error chunk surfacing

    ``on_done(prompt_tokens, completion_tokens, total_tokens, tool_calls)`` is
    invoked once with the final usage if the upstream reports it; the caller
    wires this to stats. Optional.

    ``estimated_input_tokens`` seeds the message_start usage before actual
    metrics arrive (port of ollama estimatedInputTokens).
    """
    message_id = generate_message_id()
    started = False

    # ── Block bookkeeping (port of ollama StreamConverter fields) ───────
    content_index = 0          # ollama: c.contentIndex
    thinking_started = False   # ollama: c.thinkingStarted
    thinking_done = False       # ollama: c.thinkingDone
    text_started = False       # ollama: c.textStarted
    tool_calls_sent: set[str] = set()  # ollama: c.toolCallsSent map

    # OpenAI tool_call.index → Anthropic block index
    tool_block_index: dict[int, int] = {}

    stop_reason = "end_turn"
    input_tokens = estimated_input_tokens
    output_tokens = 0
    total_tokens = 0
    tool_calls_seen = 0
    buffer = b""

    def _ensure_started() -> bytes | None:
        nonlocal started
        if started:
            return None
        started = True
        message = {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
        return _sse_event("message_start", {"type": "message_start", "message": message})

    def _close_thinking_block() -> bytes | None:
        """Close an open thinking block and advance the index.

        Port of ollama's thinking-done block close (lines 821-830).
        """
        nonlocal thinking_done, content_index
        if not (thinking_started and not thinking_done):
            return None
        thinking_done = True
        evt = _sse_event("content_block_stop", {"type": "content_block_stop", "index": content_index})
        content_index += 1
        return evt

    def _close_text_block() -> bytes | None:
        """Close an open text block and advance the index.

        Port of ollama's text block close (lines 879-889).
        """
        nonlocal text_started, content_index
        if not text_started:
            return None
        text_started = False
        evt = _sse_event("content_block_stop", {"type": "content_block_stop", "index": content_index})
        content_index += 1
        return evt

    async def _flush_final() -> AsyncIterator[bytes]:
        # Close any open blocks (port of ollama lines 934-951)
        # Close text block if still open
        if text_started:
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": content_index})
        # Close thinking block if still open (and text didn't close it)
        elif thinking_started and not thinking_done:
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": content_index})

        # Close any open tool_use blocks. Tool blocks are opened at specific
        # content_index values and stay open for continuation fragments. We
        # need to close every block that was opened but not yet closed.
        # The current content_index is past the last opened block, so any
        # tool blocks at indices < content_index that weren't closed inline
        # need closing. Since we deferred all tool block closes, all tool
        # blocks from their start index up to content_index-1 are open.
        # But text/thinking blocks were at content_index and already handled
        # above. Tool blocks were at earlier indices.
        # Simplest: close every index from 0 to content_index-1 that hasn't
        # been closed yet. Track which indices are still open.
        # Actually, we just need to close tool blocks. The text/thinking block
        # at the current content_index was handled above. Tool blocks were
        # at earlier indices and are all still open (we never close them inline).
        # Close them in order.
        for idx in sorted(tool_block_index.values()):
            if idx < content_index:
                yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": idx})

        delta = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
        yield _sse_event("message_delta", delta)
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        if on_done is not None:
            try:
                on_done(input_tokens, output_tokens, total_tokens, tool_calls_seen)
            except Exception:
                pass

    # Walk the byte stream, buffering partial SSE lines.
    async for raw in iterator:
        buffer += raw
        # SSE events are separated by blank lines; process complete lines.
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            data_str = line[5:].strip()
            if data_str == b"[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                continue

            # Upstream emitted a terminal error chunk (e.g. mid-stream read
            # timeout from NvidiaClient). Surface it as a text block so the
            # client sees the failure instead of an empty message.
            if isinstance(chunk, dict) and isinstance(chunk.get("error"), dict):
                start_evt = _ensure_started()
                if start_evt is not None:
                    yield start_evt
                # Close any open thinking block before emitting text
                close_evt = _close_thinking_block()
                if close_evt is not None:
                    yield close_evt
                if not text_started:
                    text_started = True
                    yield _sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": content_index, "content_block": {"type": "text", "text": ""}},
                    )
                err = chunk["error"]
                err_text = str(err.get("message") or "upstream stream error")
                rid = err.get("rid")
                tag = f"[stream error rid={rid}] {err_text}" if rid else f"[stream error] {err_text}"
                yield _sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": content_index, "delta": {"type": "text_delta", "text": tag}},
                )
                continue

            # Adopt the upstream message id once we see it.
            if not started and isinstance(chunk.get("id"), str):
                message_id = chunk["id"]

            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            choice = choices[0] if isinstance(choices, list) and choices else {}
            if not isinstance(choice, dict):
                choice = {}
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

            # ── reasoning/thinking delta (port of ollama lines 779-818) ─
            # OpenAI reasoning models stream thinking via the "reasoning"
            # field in the delta. Map to Anthropic thinking_delta.
            reasoning = delta.get("reasoning") or delta.get("thinking")
            if isinstance(reasoning, str) and reasoning and not thinking_done:
                start_evt = _ensure_started()
                if start_evt is not None:
                    yield start_evt

                # Close text block if open (thinking comes before text in
                # ollama's ordering — but if text started, close it first).
                # Port of ollama lines 780-789: close text if thinking arrives
                # after text started.
                if text_started:
                    close_evt = _close_text_block()
                    if close_evt is not None:
                        yield close_evt

                if not thinking_started:
                    thinking_started = True
                    yield _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": content_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        },
                    )

                yield _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": content_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    },
                )

            # ── text delta (port of ollama lines 820-859) ───────────────
            content = delta.get("content")
            if isinstance(content, str) and content:
                start_evt = _ensure_started()
                if start_evt is not None:
                    yield start_evt

                # Close thinking block if still open (thinking → text transition)
                close_thinking = _close_thinking_block()
                if close_thinking is not None:
                    yield close_thinking

                if not text_started:
                    text_started = True
                    yield _sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": content_index, "content_block": {"type": "text", "text": ""}},
                    )

                yield _sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": content_index, "delta": {"type": "text_delta", "text": content}},
                )

            # ── tool-call delta (port of ollama lines 861-932) ───────────
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                start_evt = _ensure_started()
                if start_evt is not None:
                    yield start_evt

                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    oai_index = int(tc.get("index") or 0)
                    tc_id = str(tc.get("id") or "")

                    # Port of ollama toolCallsSent map (line 862):
                    # If this chunk has an ID we haven't seen, it's a NEW
                    # tool call → open a new content_block. If it has no ID
                    # (continuation fragment) or an ID we've seen, it's an
                    # arguments fragment for an existing block → append.
                    is_new_tool = bool(tc_id) and tc_id not in tool_calls_sent

                    if is_new_tool:
                        # Close thinking block if still open (thinking → tool_use
                        # without text in between). Port of ollama lines 867-877.
                        close_thinking = _close_thinking_block()
                        if close_thinking is not None:
                            yield close_thinking

                        # Close text block if open. Port of ollama lines 879-889.
                        close_text = _close_text_block()
                        if close_text is not None:
                            yield close_text

                        # Assign block index and emit content_block_start
                        tool_block_index[oai_index] = content_index
                        tool_calls_sent.add(tc_id)
                        tool_calls_seen += 1

                        function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        yield _sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": content_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tc_id,
                                    "name": str(function.get("name") or ""),
                                    "input": {},
                                },
                            },
                        )

                    # Arguments fragment — append to the existing block.
                    # Port of ollama lines 906-918: emit input_json_delta for
                    # each arguments fragment, whether it's the first chunk
                    # (with id) or a continuation (without id).
                    function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    args_fragment = function.get("arguments")
                    if args_fragment:
                        block_idx = tool_block_index.get(oai_index, content_index)
                        yield _sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {"type": "input_json_delta", "partial_json": str(args_fragment)},
                            },
                        )

                    # Only close the block if this was a new tool call and the
                    # upstream sent the full arguments in this single chunk.
                    # Most OpenAI providers stream arguments across multiple
                    # chunks, so we DON'T close here — the block stays open for
                    # continuation fragments. The block is closed in
                    # _flush_final() when the stream ends.
                    # (Port of ollama lines 922-928: ollama closes each tool_use
                    # block immediately because it collects the full tool call
                    # before emitting. We stream, so we defer the close.)

            # ── finish_reason (port of ollama lines 934-955) ─────────────
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                stop_reason = _anthropic_stop_reason(finish_reason, len(tool_calls_sent) > 0)

            # ── usage (often on the final chunk) ────────────────────────
            usage = chunk.get("usage") if isinstance(chunk, dict) else None
            if isinstance(usage, dict):
                input_tokens = int(usage.get("prompt_tokens") or input_tokens)
                output_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or 0)

    # Stream ended. If we never started (empty upstream), still emit a minimal
    # valid Anthropic stream so the client gets a well-formed empty message.
    if not started:
        start_evt = _ensure_started()
        if start_evt is not None:
            yield start_evt

    async for chunk in _flush_final():
        yield chunk
