from __future__ import annotations

import json
import random
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any


# ── OpenAI Responses API ─────────────────────────────────────────────────
#
# Ported from Ollama's openai/responses.go. Converts Responses API requests
# to internal chat format, routes through NVIDIA, converts responses back.
# Supports both non-streaming and streaming (SSE) modes.

class ResponsesInput:
    """Discriminated union for Responses API input: string or array of items."""
    def __init__(self, data: Any):
        self.text = ""
        self.items: list[Any] = []
        if isinstance(data, str):
            self.text = data
        elif isinstance(data, list):
            self.items = data
        else:
            raise ValueError("input must be string or array")


class ResponsesReasoning:
    def __init__(self, effort: str = "", summary: str = ""):
        self.effort = effort
        self.summary = summary


class ResponsesTextFormat:
    def __init__(self, format_type: str = "text", name: str = "", schema: Any = None, strict: bool | None = None):
        self.type = format_type
        self.name = name
        self.schema = schema
        self.strict = strict


class ResponsesText:
    def __init__(self, format: ResponsesTextFormat | None = None):
        self.format = format


class ResponsesTool:
    def __init__(self, name: str, description: str = "", parameters: dict | None = None, strict: bool = False):
        self.type = "function"
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.strict = strict


class ResponsesRequest:
    def __init__(
        self,
        model: str,
        input: ResponsesInput,
        instructions: str = "",
        max_output_tokens: int | None = None,
        reasoning: ResponsesReasoning | None = None,
        text: ResponsesText | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        truncation: str | None = None,
        tools: list[ResponsesTool] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        background: bool = False,
    ):
        self.model = model
        self.input = input
        self.instructions = instructions
        self.max_output_tokens = max_output_tokens
        self.reasoning = reasoning or ResponsesReasoning()
        self.text = text
        self.top_p = top_p
        self.temperature = temperature
        self.truncation = truncation
        self.tools = tools or []
        self.tool_choice = tool_choice
        self.stream = stream
        self.background = background


def responses_request_to_openai(req: ResponsesRequest) -> dict[str, Any]:
    """Convert ResponsesRequest to OpenAI chat completions payload."""
    messages: list[dict[str, Any]] = []

    # Add instructions as system message if present
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})

    # Handle simple string input
    if req.input.text:
        messages.append({"role": "user", "content": req.input.text})

    # Handle array of input items
    pending_thinking = ""
    for item in req.input.items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")
        if item_type == "message":
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Flatten content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "input_text":
                        text_parts.append(str(block.get("text") or ""))
                if text_parts:
                    messages.append({"role": role, "content": "\n".join(text_parts)})
        elif item_type == "reasoning":
            # Store thinking to merge with next assistant message
            pending_thinking = item.get("encrypted_content", "")
        elif item_type == "function_call":
            # Convert function call to assistant message with tool_calls
            function = item.get("name", "")
            arguments = item.get("arguments", "{}")
            call_id = item.get("call_id", f"call_{len(messages)}")
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {"name": function, "arguments": arguments},
            }
            # Merge into existing assistant message if present
            if messages and messages[-1].get("role") == "assistant":
                existing_calls = messages[-1].get("tool_calls", [])
                existing_calls.append(tool_call)
                messages[-1]["tool_calls"] = existing_calls
                if pending_thinking:
                    messages[-1]["thinking"] = pending_thinking
                    pending_thinking = ""
            else:
                msg = {"role": "assistant", "tool_calls": [tool_call]}
                if pending_thinking:
                    msg["thinking"] = pending_thinking
                    pending_thinking = ""
                messages.append(msg)
        elif item_type == "function_call_output":
            # Convert function call output to tool message
            call_id = item.get("call_id", "call_0")
            output = item.get("output", "")
            if isinstance(output, list):
                text_parts = []
                for block in output:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text_parts.append(str(block.get("text") or ""))
                output = "\n".join(text_parts)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output,
            })

    # If there's trailing reasoning without a following message, emit it
    if pending_thinking:
        messages.append({"role": "assistant", "thinking": pending_thinking})

    # Build options
    options: dict[str, Any] = {}
    if req.temperature is not None:
        options["temperature"] = req.temperature
    else:
        options["temperature"] = 1.0

    if req.top_p is not None:
        options["top_p"] = req.top_p
    else:
        options["top_p"] = 1.0

    if req.max_output_tokens is not None:
        options["max_tokens"] = req.max_output_tokens

    # Reasoning
    think = None
    if req.reasoning.effort:
        if req.reasoning.effort == "none":
            think = {"value": False}
        else:
            think = {"value": req.reasoning.effort}

    # Tools
    tools = []
    for t in req.tools:
        tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        })

    tool_choice = req.tool_choice
    if tools and not tool_choice:
        tool_choice = "auto"

    # Text format
    format_json = None
    if req.text and req.text.format and req.text.format.type == "json_schema":
        format_json = req.text.format.schema

    payload = {
        "model": req.model,
        "messages": messages,
        "stream": req.stream,
        "temperature": options.get("temperature", 0.7),
        "top_p": options.get("top_p", 1.0),
    }
    if req.max_output_tokens is not None:
        payload["max_tokens"] = req.max_output_tokens
    if think:
        payload["reasoning"] = {"effort": think["value"]} if isinstance(think["value"], str) else {"effort": "medium"}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    if format_json:
        payload["response_format"] = {"type": "json_schema", "json_schema": format_json}

    return sanitize_openai_payload(payload)


def responses_response_from_openai(model: str, response_id: str, item_id: str, openai_response: dict[str, Any], request: ResponsesRequest) -> dict[str, Any]:
    """Convert OpenAI chat completion response to Responses API format."""
    choices = openai_response.get("choices") or []
    if not choices:
        return {}

    choice = choices[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason")

    output: list[dict[str, Any]] = []
    output_index = 0

    # Add reasoning item if thinking present
    thinking = message.get("thinking", "")
    if thinking:
        reasoning_id = f"rs_{response_id}"
        output.append({
            "id": reasoning_id,
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": thinking}],
            "encrypted_content": thinking,
        })
        output_index += 1

    # Handle tool calls
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            function = tc.get("function") or {}
            fc_id = f"fc_{response_id}_{i}"
            output.append({
                "id": fc_id,
                "type": "function_call",
                "status": "completed",
                "call_id": str(tc.get("id") or f"call_{i}"),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or "{}"),
            })
            output_index += 1
    else:
        # Text content
        content = message.get("content", "")
        if content:
            output.append({
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": content,
                    "annotations": [],
                    "logprobs": [],
                }],
            })
            output_index += 1

    # Build usage
    usage_data = openai_response.get("usage") or {}
    usage = {
        "input_tokens": int(usage_data.get("prompt_tokens") or 0),
        "output_tokens": int(usage_data.get("completion_tokens") or 0),
        "total_tokens": int(usage_data.get("total_tokens") or 0),
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }

    # Build tools list for response
    tools = []
    for t in request.tools:
        tools.append({
            "type": "function",
            "name": t.name,
            "description": t.description,
            "strict": t.strict,
            "parameters": t.parameters,
        })
    if not tools:
        tools = []

    # Text format
    text_format = {"type": "text"}
    if request.text and request.text.format:
        text_format = {"type": request.text.format.type}
        if request.text.format.name:
            text_format["name"] = request.text.format.name
        if request.text.format.schema:
            text_format["schema"] = request.text.format.schema
        if request.text.format.strict is not None:
            text_format["strict"] = request.text.format.strict

    # Reasoning
    reasoning = None
    if request.reasoning.effort or request.reasoning.summary:
        reasoning = {}
        if request.reasoning.effort:
            reasoning["effort"] = request.reasoning.effort
        if request.reasoning.summary:
            reasoning["summary"] = request.reasoning.summary

    # Truncation
    truncation = "disabled"
    if request.truncation:
        truncation = request.truncation

    # Temperature/top_p
    temperature = 1.0
    if request.temperature is not None:
        temperature = request.temperature

    top_p = 1.0
    if request.top_p is not None:
        top_p = request.top_p

    completed_at = int(time.time()) if finish_reason else None

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "completed_at": completed_at,
        "status": "completed" if finish_reason else "in_progress",
        "incomplete_details": {"reason": finish_reason} if finish_reason == "length" else None,
        "model": model,
        "previous_response_id": None,
        "instructions": request.instructions if request.instructions else None,
        "output": output,
        "error": None,
        "tools": tools,
        "tool_choice": "auto",
        "truncation": truncation,
        "parallel_tool_calls": True,
        "text": {"format": text_format},
        "top_p": top_p,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_logprobs": 0,
        "temperature": temperature,
        "reasoning": reasoning,
        "usage": usage,
        "max_output_tokens": request.max_output_tokens,
        "max_tool_calls": None,
        "store": False,
        "background": request.background,
        "service_tier": "default",
        "metadata": {},
        "safety_identifier": None,
        "prompt_cache_key": None,
    }


class ResponsesStreamConverter:
    """Converts OpenAI SSE chunks to Responses API streaming events."""

    def __init__(self, response_id: str, item_id: str, model: str, request: ResponsesRequest):
        self.response_id = response_id
        self.item_id = item_id
        self.model = model
        self.request = request

        # State tracking
        self.first_write = True
        self.output_index = 0
        self.content_index = 0
        self.content_started = False
        self.tool_calls_sent = False
        self.accumulated_text = ""
        self.sequence_number = 0

        # Reasoning/thinking state
        self.accumulated_thinking = ""
        self.reasoning_item_id = ""
        self.reasoning_started = False
        self.reasoning_done = False

        # Tool calls state
        self.tool_call_items: list[dict[str, Any]] = []

    def _new_event(self, event_type: str, data: dict[str, Any]) -> bytes:
        data["type"] = event_type
        data["sequence_number"] = self.sequence_number
        self.sequence_number += 1
        return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()

    def _build_response_object(self, status: str, output: list[dict[str, Any]], usage: dict[str, Any] | None) -> dict[str, Any]:
        """Build a full response object for streaming events."""
        instructions = self.request.instructions if self.request.instructions else None

        truncation = "disabled"
        if self.request.truncation:
            truncation = self.request.truncation

        tools = []
        for t in self.request.tools:
            tools.append({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "strict": t.strict,
                "parameters": t.parameters,
            })
        if not tools:
            tools = []

        text_format = {"type": "text"}
        if self.request.text and self.request.text.format:
            text_format = {"type": self.request.text.format.type}
            if self.request.text.format.name:
                text_format["name"] = self.request.text.format.name
            if self.request.text.format.schema:
                text_format["schema"] = self.request.text.format.schema
            if self.request.text.format.strict is not None:
                text_format["strict"] = self.request.text.format.strict

        reasoning = None
        if self.request.reasoning.effort or self.request.reasoning.summary:
            reasoning = {}
            if self.request.reasoning.effort:
                reasoning["effort"] = self.request.reasoning.effort
            if self.request.reasoning.summary:
                reasoning["summary"] = self.request.reasoning.summary

        top_p = 1.0
        if self.request.top_p is not None:
            top_p = self.request.top_p

        temperature = 1.0
        if self.request.temperature is not None:
            temperature = self.request.temperature

        return {
            "id": self.response_id,
            "object": "response",
            "created_at": int(time.time()),
            "completed_at": None,
            "status": status,
            "incomplete_details": None,
            "model": self.model,
            "previous_response_id": None,
            "instructions": instructions,
            "output": output,
            "error": None,
            "tools": tools,
            "tool_choice": "auto",
            "truncation": truncation,
            "parallel_tool_calls": True,
            "text": {"format": text_format},
            "top_p": top_p,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "top_logprobs": 0,
            "temperature": temperature,
            "reasoning": reasoning,
            "usage": usage,
            "max_output_tokens": self.request.max_output_tokens,
            "max_tool_calls": None,
            "store": False,
            "background": self.request.background,
            "service_tier": "default",
            "metadata": {},
            "safety_identifier": None,
            "prompt_cache_key": None,
        }

    def process(self, openai_chunk: dict[str, Any]) -> list[bytes]:
        """Process an OpenAI SSE chunk and return Responses API events."""
        events: list[bytes] = []

        choices = openai_chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return events

        choice = choices[0]
        if not isinstance(choice, dict):
            return events

        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}

        has_tool_calls = "tool_calls" in delta and isinstance(delta["tool_calls"], list) and delta["tool_calls"]
        has_thinking = "thinking" in delta and delta["thinking"]
        has_content = "content" in delta and delta["content"]

        # First chunk - emit initial events
        if self.first_write:
            self.first_write = False
            events.append(self._create_response_created_event())
            events.append(self._create_response_in_progress_event())

        # Handle reasoning/thinking (emitted first)
        if has_thinking:
            events.extend(self._process_thinking(delta["thinking"]))

        # Handle tool calls
        if has_tool_calls:
            events.extend(self._process_tool_calls(delta["tool_calls"]))
            self.tool_calls_sent = True

        # Handle text content (only if no tool calls)
        if not has_tool_calls and not self.tool_calls_sent and has_content:
            events.extend(self._process_text_content(delta["content"]))

        # Handle completion
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            events.extend(self._process_completion(finish_reason, openai_chunk))

        return events

    def _create_response_created_event(self) -> bytes:
        return self._new_event("response.created", {
            "response": self._build_response_object("in_progress", [], None),
        })

    def _create_response_in_progress_event(self) -> bytes:
        return self._new_event("response.in_progress", {
            "response": self._build_response_object("in_progress", [], None),
        })

    def _process_thinking(self, thinking: str) -> list[bytes]:
        events: list[bytes] = []

        # Start reasoning item if not started
        if not self.reasoning_started:
            self.reasoning_started = True
            self.reasoning_item_id = f"rs_{random.randint(100000, 999999)}"

            events.append(self._new_event("response.output_item.added", {
                "output_index": self.output_index,
                "item": {
                    "id": self.reasoning_item_id,
                    "type": "reasoning",
                    "summary": [],
                },
            }))

        # Accumulate thinking
        self.accumulated_thinking += thinking

        # Emit delta
        events.append(self._new_event("response.reasoning_summary_text.delta", {
            "item_id": self.reasoning_item_id,
            "output_index": self.output_index,
            "summary_index": 0,
            "delta": thinking,
        }))

        return events

    def _finish_reasoning(self) -> list[bytes]:
        if not self.reasoning_started or self.reasoning_done:
            return []
        self.reasoning_done = True

        events = [
            self._new_event("response.reasoning_summary_text.done", {
                "item_id": self.reasoning_item_id,
                "output_index": self.output_index,
                "summary_index": 0,
                "text": self.accumulated_thinking,
            }),
            self._new_event("response.output_item.done", {
                "output_index": self.output_index,
                "item": {
                    "id": self.reasoning_item_id,
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": self.accumulated_thinking}],
                    "encrypted_content": self.accumulated_thinking,
                },
            }),
        ]
        self.output_index += 1
        return events

    def _process_tool_calls(self, tool_calls: list[dict]) -> list[bytes]:
        events: list[bytes] = []

        # Finish reasoning first if it was started
        events.extend(self._finish_reasoning())

        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            oai_index = int(tc.get("index") or 0)

            fc_item_id = f"fc_{random.randint(100000, 999999)}_{i}"

            # Store for final output (with status: completed)
            function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            tool_call_item = {
                "id": fc_item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": str(tc.get("id") or f"call_{oai_index}"),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or ""),
            }
            self.tool_call_items.append(tool_call_item)

            # response.output_item.added for function call
            events.append(self._new_event("response.output_item.added", {
                "output_index": self.output_index + i,
                "item": {
                    "id": fc_item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": tool_call_item["call_id"],
                    "name": tool_call_item["name"],
                    "arguments": "",
                },
            }))

            # response.function_call_arguments.delta
            args = function.get("arguments", "")
            if args:
                events.append(self._new_event("response.function_call_arguments.delta", {
                    "item_id": fc_item_id,
                    "output_index": self.output_index + i,
                    "delta": args,
                }))

            # response.function_call_arguments.done
            events.append(self._new_event("response.function_call_arguments.done", {
                "item_id": fc_item_id,
                "output_index": self.output_index + i,
                "arguments": args,
            }))

            # response.output_item.done for function call
            events.append(self._new_event("response.output_item.done", {
                "output_index": self.output_index + i,
                "item": tool_call_item,
            }))

        return events

    def _process_text_content(self, content: str) -> list[bytes]:
        events: list[bytes] = []

        # Finish reasoning first if it was started
        events.extend(self._finish_reasoning())

        # Emit output item and content part for first text content
        if not self.content_started:
            self.content_started = True

            # response.output_item.added
            events.append(self._new_event("response.output_item.added", {
                "output_index": self.output_index,
                "item": {
                    "id": self.item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }))

            # response.content_part.added
            events.append(self._new_event("response.content_part.added", {
                "item_id": self.item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "part": {"type": "output_text", "text": ""},
            }))

        # Accumulate and emit delta
        self.accumulated_text += content
        events.append(self._new_event("response.output_text.delta", {
            "item_id": self.item_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "delta": content,
        }))

        return events

    def _process_completion(self, finish_reason: str, openai_chunk: dict[str, Any]) -> list[bytes]:
        events: list[bytes] = []

        # Finish any pending reasoning
        events.extend(self._finish_reasoning())

        # Finish text content if started
        if self.content_started:
            events.append(self._new_event("response.output_text.done", {
                "item_id": self.item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "text": self.accumulated_text,
            }))
            events.append(self._new_event("response.output_item.done", {
                "output_index": self.output_index,
                "item": {
                    "id": self.item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": self.accumulated_text, "annotations": []}],
                },
            }))
            self.output_index += 1

        # Build usage
        usage_data = openai_chunk.get("usage")
        usage = None
        if isinstance(usage_data, dict):
            usage = {
                "input_tokens": int(usage_data.get("prompt_tokens") or 0),
                "output_tokens": int(usage_data.get("completion_tokens") or 0),
                "total_tokens": int(usage_data.get("total_tokens") or 0),
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }

        # response.completed
        status = "completed"
        if finish_reason == "length":
            status = "incomplete"

        events.append(self._new_event("response.completed", {
            "response": self._build_response_object(status, [], usage),
        }))

        return events


def openai_error(message: str, code: str, status: int) -> dict[str, Any]:
    """OpenAI-compatible error response. Maps status to standard error types."""
    if code == "bad_request":
        error_type = "invalid_request_error"
    elif code == "not_found":
        error_type = "not_found_error"
    elif code == "rate_limit":
        error_type = "rate_limit_error"
    elif code == "unauthorized":
        error_type = "authentication_error"
    elif code == "forbidden":
        error_type = "permission_error"
    elif status >= 500:
        error_type = "api_error"
    else:
        error_type = code
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": status if isinstance(status, int) else None,
        }
    }


def anthropic_error(message: str, status: int) -> dict[str, Any]:
    """Anthropic-compatible error response for /v1/messages."""
    error_type = "api_error"
    if status == 400:
        error_type = "invalid_request_error"
    elif status == 401:
        error_type = "authentication_error"
    elif status == 403:
        error_type = "permission_error"
    elif status == 404:
        error_type = "not_found_error"
    elif status == 429:
        error_type = "rate_limit_error"
    return {"type": "error", "error": {"type": error_type, "message": message}}


def streaming_headers() -> dict[str, str]:
    """Standard headers for all SSE streaming responses."""
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


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
        normalized.append(normalized_message)
    return normalized


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


def anthropic_response(model: str, content: str) -> dict[str, Any]:
    return anthropic_response_from_blocks(model, [{"type": "text", "text": content}], "end_turn")


def anthropic_response_from_blocks(
    model: str,
    content_blocks: list[dict[str, Any]],
    stop_reason: str,
) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def anthropic_system_text(system: Any) -> str:
    if isinstance(system, list):
        return "\n".join(
            str(block.get("text") or "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(system or "")


def anthropic_messages_to_openai(body: dict[str, Any]) -> list[dict[str, Any]]:
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

        if not isinstance(content, list):
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": str(content)})
            continue

        blocks = [block for block in content if isinstance(block, dict) and block.get("type") != "thinking"]
        text_parts = [str(block.get("text") or "") for block in blocks if block.get("type") == "text"]
        tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
        tool_results = [block for block in blocks if block.get("type") == "tool_result"]

        if role == "assistant" and tool_uses:
            messages.append(
                {
                    "role": "assistant",
                    "content": "\n".join(text_parts),
                    "tool_calls": [
                        {
                            "id": str(tool_use.get("id") or f"call_{index}"),
                            "type": "function",
                            "function": {
                                "name": str(tool_use.get("name") or ""),
                                "arguments": json.dumps(tool_use.get("input") or {}),
                            },
                        }
                        for index, tool_use in enumerate(tool_uses)
                    ],
                }
            )
            continue

        if tool_results:
            for result in tool_results:
                result_content = result.get("content", "")
                if isinstance(result_content, list):
                    result_content = "\n".join(
                        str(block.get("text") or "")
                        for block in result_content
                        if isinstance(block, dict)
                    )
                elif not isinstance(result_content, str):
                    result_content = json.dumps(result_content)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(result.get("tool_use_id") or "call_0"),
                        "content": result_content,
                    }
                )
            visible_text = [text for text in text_parts if not text.strip().startswith("<system-reminder")]
            if visible_text:
                messages.append({"role": "user", "content": visible_text[-1].strip()})
            continue

        messages.append({"role": "assistant" if role == "assistant" else "user", "content": "\n".join(text_parts)})

    return messages


def anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    openai_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool["name"]),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema") or {},
                },
            }
        )
    return openai_tools


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": str(tool_choice["name"])}}
    return None


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
    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = anthropic_tool_choice_to_openai(body.get("tool_choice")) or "auto"
    # Sanitize (clamp sampling params, drop unsupported fields) before the
    # payload hits NVIDIA. Anthropic clients can still send out-of-range
    # temperature; clamp it instead of letting NVIDIA 400 the request.
    return sanitize_openai_payload(payload)


def openai_response_to_anthropic(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices") or []
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    content_blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        content_blocks.append({"type": "text", "text": content})

    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            tool_input = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": str(tool_call.get("id") or f"call_{len(content_blocks)}"),
                "name": str(function.get("name") or ""),
                "input": tool_input if isinstance(tool_input, dict) else {},
            }
        )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": extract_router_content(payload)})

    finish_reason = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
    stop_reason = "tool_use" if finish_reason == "tool_calls" or any(block["type"] == "tool_use" for block in content_blocks) else "end_turn"
    response = anthropic_response_from_blocks(model, content_blocks, stop_reason)
    usage = payload.get("usage") or {}
    response["usage"] = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }
    return response


async def anthropic_sse_from_response(response: dict[str, Any]) -> AsyncIterator[bytes]:
    message = {**response, "content": []}
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': message}, separators=(',', ':'))}\n\n".encode()

    for index, block in enumerate(response.get("content") or []):
        start = {"type": "content_block_start", "index": index, "content_block": block}
        if block.get("type") == "text":
            start["content_block"] = {"type": "text", "text": ""}
        elif block.get("type") == "tool_use":
            start["content_block"] = {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": {},
            }
        yield f"event: content_block_start\ndata: {json.dumps(start, separators=(',', ':'))}\n\n".encode()

        if block.get("type") == "text" and block.get("text"):
            delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": block["text"]},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta, separators=(',', ':'))}\n\n".encode()
        elif block.get("type") == "tool_use":
            delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta, separators=(',', ':'))}\n\n".encode()

        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': index}, separators=(',', ':'))}\n\n".encode()

    delta = {
        "type": "message_delta",
        "delta": {"stop_reason": response.get("stop_reason") or "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": response.get("usage", {}).get("output_tokens", 0)},
    }
    yield f"event: message_delta\ndata: {json.dumps(delta, separators=(',', ':'))}\n\n".encode()
    yield b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"


async def anthropic_sse(model: str, content: str) -> AsyncIterator[bytes]:
    async for chunk in anthropic_sse_from_response(anthropic_response(model, content)):
        yield chunk


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    """Encode one Anthropic SSE event as bytes: 'event: <e>\\ndata: <json>\\n\\n'."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _anthropic_stop_reason(finish_reason: str | None) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    return {
        "tool_calls": "tool_use",
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }.get(finish_reason or "", "end_turn")


async def openai_sse_to_anthropic_sse(
    iterator: AsyncIterator[bytes],
    model: str,
    on_done: Any = None,
    on_worker_limit: Any = None,
) -> AsyncIterator[bytes]:
    """Translate an OpenAI/NVIDIA SSE byte stream into Anthropic SSE events.

    Consumes raw bytes from NvidiaClient.stream_chat() and yields Anthropic
    event bytes as they arrive — message_start, content_block_start/delta/stop,
    message_delta, message_stop. Real streaming, no buffering of the full body.

    `on_done(prompt_tokens, completion_tokens, total_tokens, tool_calls)` is
    invoked once with the final usage if the upstream reports it; the caller
    wires this to stats. Optional.

    `on_worker_limit()` is invoked when a worker concurrency limit error is
    detected in an SSE error chunk. This allows the caller to cool the key
    and trigger failover. Optional.
    """
    message_id = f"msg_{uuid.uuid4().hex}"
    started = False
    # Content-block bookkeeping. Anthropic blocks are indexed in emission
    # order: a text block (if any) at index 0, then tool_use blocks after.
    text_block_open = False
    # openai tool_call.index -> anthropic block index
    tool_block_index: dict[int, int] = {}
    next_block_index = 0  # next anthropic block index to hand out
    stop_reason = "end_turn"
    input_tokens = 0
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
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return _sse_event("message_start", {"type": "message_start", "message": message})

    def _close_all_blocks() -> list[bytes]:
        """Emit content_block_stop for every currently-open block."""
        out: list[bytes] = []
        # Text block is index 0; tool blocks are the rest, in order.
        indices = sorted(tool_block_index.values())
        if text_block_open:
            indices = [0] + indices
        for idx in indices:
            out.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": idx}))
        return out

    async def _flush_final() -> AsyncIterator[bytes]:
        # Close any open blocks, then message_delta + message_stop.
        for chunk in _close_all_blocks():
            yield chunk
        delta = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
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
                if not text_block_open:
                    text_block_open = True
                    next_block_index = 1
                    yield _sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    )
                err = chunk["error"]
                err_text = str(err.get("message") or "upstream stream error")
                # Detect worker concurrency limit in stream errors too
                if "Worker local total request limit reached" in err_text:
                    err_text = "upstream worker concurrency limit reached"
                    if on_worker_limit is not None:
                        try:
                            on_worker_limit()
                        except Exception:
                            pass
                rid = err.get("rid")
                tag = f"[stream error rid={rid}] {err_text}" if rid else f"[stream error] {err_text}"
                yield _sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": tag}},
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

            # ── text delta ──────────────────────────────────────────────
            content = delta.get("content")
            if isinstance(content, str) and content:
                start_evt = _ensure_started()
                if start_evt is not None:
                    yield start_evt
                if not text_block_open:
                    text_block_open = True
                    next_block_index = 1  # text is index 0; tools come after
                    yield _sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    )
                yield _sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": content}},
                )

            # ── tool-call delta ─────────────────────────────────────────
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                start_evt = _ensure_started()
                if start_evt is not None:
                    yield start_evt
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    oai_index = int(tc.get("index") or 0)
                    if oai_index not in tool_block_index:
                        tool_block_index[oai_index] = next_block_index
                        next_block_index += 1
                        tool_calls_seen += 1
                        function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        yield _sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": tool_block_index[oai_index],
                                "content_block": {
                                    "type": "tool_use",
                                    "id": str(tc.get("id") or f"call_{oai_index}"),
                                    "name": str(function.get("name") or ""),
                                    "input": {},
                                },
                            },
                        )
                    function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    args_fragment = function.get("arguments")
                    if args_fragment:
                        yield _sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": tool_block_index[oai_index],
                                "delta": {"type": "input_json_delta", "partial_json": str(args_fragment)},
                            },
                        )

            # ── finish_reason ───────────────────────────────────────────
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                stop_reason = _anthropic_stop_reason(finish_reason)

            # ── usage (often on the final chunk) ────────────────────────
            usage = chunk.get("usage") if isinstance(chunk, dict) else None
            if isinstance(usage, dict):
                input_tokens = int(usage.get("prompt_tokens") or 0)
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


# ── OpenAI Responses API ─────────────────────────────────────────────────
#
# Ported from Ollama's openai/responses.go. Converts Responses API requests
# to internal chat format, routes through NVIDIA, converts responses back.
# Supports both non-streaming and streaming (SSE) modes.

class ResponsesInput:
    """Discriminated union for Responses API input: string or array of items."""
    def __init__(self, data: Any):
        self.text = ""
        self.items: list[Any] = []
        if isinstance(data, str):
            self.text = data
        elif isinstance(data, list):
            self.items = data
        else:
            raise ValueError("input must be string or array")


class ResponsesReasoning:
    def __init__(self, effort: str = "", summary: str = ""):
        self.effort = effort
        self.summary = summary


class ResponsesTextFormat:
    def __init__(self, format_type: str = "text", name: str = "", schema: Any = None, strict: bool | None = None):
        self.type = format_type
        self.name = name
        self.schema = schema
        self.strict = strict


class ResponsesText:
    def __init__(self, format: ResponsesTextFormat | None = None):
        self.format = format


class ResponsesTool:
    def __init__(self, name: str, description: str = "", parameters: dict | None = None, strict: bool = False):
        self.type = "function"
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.strict = strict


class ResponsesRequest:
    def __init__(
        self,
        model: str,
        input: ResponsesInput,
        instructions: str = "",
        max_output_tokens: int | None = None,
        reasoning: ResponsesReasoning | None = None,
        temperature: float | None = None,
        text: ResponsesText | None = None,
        top_p: float | None = None,
        truncation: str | None = None,
        tools: list[ResponsesTool] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        background: bool = False,
        include: list[str] | None = None,
        conversation: Any = None,
    ):
        self.model = model
        self.input = input
        self.instructions = instructions
        self.max_output_tokens = max_output_tokens
        self.reasoning = reasoning or ResponsesReasoning()
        self.temperature = temperature
        self.text = text
        self.top_p = top_p
        self.truncation = truncation
        self.tools = tools or []
        self.tool_choice = tool_choice
        self.stream = stream
        self.background = background
        self.include = include or []
        self.conversation = conversation


def responses_request_from_dict(data: dict[str, Any]) -> ResponsesRequest:
    """Parse a Responses API request dict into a ResponsesRequest."""
    model = str(data.get("model") or "")
    input_data = data.get("input", "")
    input_obj = ResponsesInput(input_data)

    instructions = str(data.get("instructions") or "")
    max_output_tokens = data.get("max_output_tokens")
    if max_output_tokens is not None:
        max_output_tokens = int(max_output_tokens)

    reasoning_data = data.get("reasoning", {})
    reasoning = ResponsesReasoning(
        effort=str(reasoning_data.get("effort") or ""),
        summary=str(reasoning_data.get("summary") or ""),
    )

    temperature = data.get("temperature")
    if temperature is not None:
        temperature = float(temperature)

    text_data = data.get("text")
    text = None
    if text_data and isinstance(text_data, dict):
        format_data = text_data.get("format")
        fmt = None
        if format_data and isinstance(format_data, dict):
            fmt = ResponsesTextFormat(
                format_type=str(format_data.get("type") or "text"),
                name=str(format_data.get("name") or ""),
                schema=format_data.get("schema"),
                strict=format_data.get("strict"),
            )
        text = ResponsesText(format=fmt)

    top_p = data.get("top_p")
    if top_p is not None:
        top_p = float(top_p)

    truncation = data.get("truncation")
    if truncation is not None:
        truncation = str(truncation)

    tools = []
    for t in data.get("tools", []):
        if isinstance(t, dict) and t.get("type") == "function":
            fn = t.get("function", {})
            tools.append(ResponsesTool(
                name=str(fn.get("name") or ""),
                description=str(fn.get("description") or ""),
                parameters=fn.get("parameters"),
                strict=bool(fn.get("strict", False)),
            ))

    tool_choice = data.get("tool_choice")
    stream = bool(data.get("stream", False))
    background = bool(data.get("background", False))
    include = data.get("include", [])
    if not isinstance(include, list):
        include = []
    conversation = data.get("conversation")

    return ResponsesRequest(
        model=model,
        input=input_obj,
        instructions=instructions,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        temperature=temperature,
        text=text,
        top_p=top_p,
        truncation=truncation,
        tools=tools,
        tool_choice=tool_choice,
        stream=stream,
        background=background,
        include=include,
        conversation=conversation,
    )


def responses_tools_to_openai(tools: list[ResponsesTool]) -> list[dict[str, Any]]:
    """Convert Responses API tools to OpenAI tools format."""
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        })
    return openai_tools


def responses_tool_choice_to_openai(tool_choice: Any) -> Any:
    """Convert Responses API tool_choice to OpenAI format."""
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "required":
        return "required"
    if choice_type == "function" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": str(tool_choice["name"])}}
    return None


def responses_request_to_chat_payload(req: ResponsesRequest, upstream_model: str) -> dict[str, Any]:
    """Convert ResponsesRequest to OpenAI chat completions payload for NVIDIA."""
    messages: list[dict[str, Any]] = []

    # Add instructions as system message
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})

    # Handle simple string input
    if req.input.text:
        messages.append({"role": "user", "content": req.input.text})

    # Handle array of input items
    pending_thinking = ""
    for item in req.input.items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")

        if item_type == "message":
            role = str(item.get("role") or "user")
            content = item.get("content", "")
            if isinstance(content, list):
                # Convert content blocks to text
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("input_text", "output_text"):
                        text_parts.append(str(block.get("text") or ""))
                content = "\n".join(text_parts)
            elif not isinstance(content, str):
                content = str(content)

            if role == "assistant" and pending_thinking:
                messages.append({"role": "assistant", "content": content, "thinking": pending_thinking})
                pending_thinking = ""
            else:
                messages.append({"role": role, "content": content})

        elif item_type == "function_call":
            # Convert function call to assistant message with tool_calls
            call_id = str(item.get("call_id") or item.get("id") or "")
            name = str(item.get("name") or "")
            arguments = item.get("arguments", "{}")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)

            if messages and messages[-1].get("role") == "assistant":
                # Merge into existing assistant message
                if "tool_calls" not in messages[-1]:
                    messages[-1]["tool_calls"] = []
                messages[-1]["tool_calls"].append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
                if pending_thinking:
                    messages[-1]["thinking"] = pending_thinking
                    pending_thinking = ""
            else:
                msg = {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }],
                }
                if pending_thinking:
                    msg["thinking"] = pending_thinking
                    pending_thinking = ""
                messages.append(msg)

        elif item_type == "function_call_output":
            # Convert function call output to tool message
            call_id = str(item.get("call_id") or "")
            output = item.get("output", "")
            if isinstance(output, list):
                # Convert output items to text
                text_parts = []
                for block in output:
                    if isinstance(block, dict) and block.get("type") in ("output_text", "input_text"):
                        text_parts.append(str(block.get("text") or ""))
                output = "\n".join(text_parts)
            elif not isinstance(output, str):
                output = json.dumps(output)

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output,
            })

        elif item_type == "reasoning":
            # Store reasoning to attach to next assistant message
            pending_thinking = str(item.get("encrypted_content") or "")

    # If there's trailing reasoning without a following message, emit it
    if pending_thinking:
        messages.append({"role": "assistant", "thinking": pending_thinking})

    # Build options
    options: dict[str, Any] = {}
    if req.temperature is not None:
        options["temperature"] = req.temperature
    else:
        options["temperature"] = 1.0

    if req.top_p is not None:
        options["top_p"] = req.top_p

    if req.max_output_tokens is not None:
        options["max_tokens"] = req.max_output_tokens

    # Reasoning
    think = None
    if req.reasoning.effort:
        effort = req.reasoning.effort
        if effort == "none":
            think = {"value": False}
        elif effort in ("low", "medium", "high", "max"):
            think = {"value": effort}

    # Tools
    openai_tools = responses_tools_to_openai(req.tools)
    tool_choice = responses_tool_choice_to_openai(req.tool_choice)
    if openai_tools and tool_choice is None:
        tool_choice = "auto"

    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "stream": req.stream,
    }
    if options:
        payload.update(options)
    if think is not None:
        payload["think"] = think
    if openai_tools:
        payload["tools"] = openai_tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    # Sanitize for NVIDIA
    return sanitize_openai_payload(payload)


# Response types for non-streaming
class ResponsesUsage:
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.input_tokens_details = {"cached_tokens": 0}
        self.output_tokens_details = {"reasoning_tokens": 0}


class ResponsesOutputContent:
    def __init__(self, text: str, annotations: list | None = None):
        self.type = "output_text"
        self.text = text
        self.annotations = annotations or []


class ResponsesOutputItem:
    def __init__(
        self,
        item_id: str,
        item_type: str,
        status: str = "completed",
        role: str | None = None,
        content: list[ResponsesOutputContent] | None = None,
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
        summary: list[dict] | None = None,
        encrypted_content: str | None = None,
    ):
        self.id = item_id
        self.type = item_type
        self.status = status
        self.role = role
        self.content = content
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
        self.summary = summary
        self.encrypted_content = encrypted_content


class ResponsesResponse:
    def __init__(
        self,
        response_id: str,
        model: str,
        output: list[ResponsesOutputItem],
        usage: ResponsesUsage | None = None,
        status: str = "completed",
        instructions: str | None = None,
        tools: list | None = None,
        tool_choice: str = "auto",
        truncation: str = "disabled",
        parallel_tool_calls: bool = True,
        text_format: dict | None = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
        reasoning: dict | None = None,
        max_output_tokens: int | None = None,
        completed_at: int | None = None,
    ):
        self.id = response_id
        self.object = "response"
        self.created_at = int(time.time())
        self.completed_at = completed_at
        self.status = status
        self.incomplete_details = None
        self.model = model
        self.previous_response_id = None
        self.instructions = instructions
        self.output = output
        self.error = None
        self.tools = tools or []
        self.tool_choice = tool_choice
        self.truncation = truncation
        self.parallel_tool_calls = parallel_tool_calls
        self.text = {"format": text_format or {"type": "text"}}
        self.top_p = top_p
        self.presence_penalty = 0
        self.frequency_penalty = 0
        self.top_logprobs = 0
        self.temperature = temperature
        self.reasoning = reasoning
        self.usage = usage
        self.max_output_tokens = max_output_tokens
        self.max_tool_calls = None
        self.store = False
        self.background = False
        self.service_tier = "default"
        self.metadata = {}
        self.safety_identifier = None
        self.prompt_cache_key = None

    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if v is not None}
        # Convert output items to dicts
        if "output" in data:
            data["output"] = [item.__dict__ for item in data["output"]]
        if "usage" in data and data["usage"]:
            data["usage"] = data["usage"].__dict__
        if "reasoning" in data and data["reasoning"]:
            data["reasoning"] = data["reasoning"]
        return data


def openai_response_to_responses(model: str, response_id: str, item_id: str, openai_payload: dict[str, Any], request: ResponsesRequest) -> ResponsesResponse:
    """Convert NVIDIA OpenAI response to Responses API response."""
    choices = openai_payload.get("choices", [])
    if not choices:
        return ResponsesResponse(
            response_id=response_id,
            model=model,
            output=[],
            usage=ResponsesUsage(),
        )

    first = choices[0]
    message = first.get("message", {})
    finish_reason = first.get("finish_reason")

    output: list[ResponsesOutputItem] = []
    usage_data = openai_payload.get("usage", {})
    usage = ResponsesUsage(
        input_tokens=int(usage_data.get("prompt_tokens") or 0),
        output_tokens=int(usage_data.get("completion_tokens") or 0),
        total_tokens=int(usage_data.get("total_tokens") or 0),
    )

    # Handle reasoning/thinking
    thinking = message.get("thinking", "")
    if thinking:
        output.append(ResponsesOutputItem(
            item_id=f"rs_{response_id}",
            item_type="reasoning",
            summary=[{"type": "summary_text", "text": thinking}],
            encrypted_content=thinking,
        ))

    # Handle tool calls
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            output.append(ResponsesOutputItem(
                item_id=f"fc_{response_id}_{i}",
                item_type="function_call",
                call_id=str(tc.get("id") or f"call_{i}"),
                name=str(fn.get("name") or ""),
                arguments=json.dumps(fn.get("arguments") or {}),
            ))
    else:
        # Regular text message
        content = message.get("content", "")
        if content:
            output.append(ResponsesOutputItem(
                item_id=item_id,
                item_type="message",
                role="assistant",
                content=[ResponsesOutputContent(text=content)],
            ))

    # Build reasoning output
    reasoning_out = None
    if request.reasoning.effort or request.reasoning.summary:
        reasoning_out = {}
        if request.reasoning.effort:
            reasoning_out["effort"] = request.reasoning.effort
        if request.reasoning.summary:
            reasoning_out["summary"] = request.reasoning.summary

    # Build tools list
    tools = []
    for t in request.tools:
        tools.append({
            "type": "function",
            "name": t.name,
            "description": t.description,
            "strict": t.strict,
            "parameters": t.parameters,
        })

    # Text format
    text_format = {"type": "text"}
    if request.text and request.text.format:
        text_format = {"type": request.text.format.type}
        if request.text.format.name:
            text_format["name"] = request.text.format.name
        if request.text.format.schema:
            text_format["schema"] = request.text.format.schema
        if request.text.format.strict is not None:
            text_format["strict"] = request.text.format.strict

    return ResponsesResponse(
        response_id=response_id,
        model=model,
        output=output,
        usage=usage,
        status="completed" if finish_reason != "length" else "incomplete",
        instructions=request.instructions if request.instructions else None,
        tools=tools,
        tool_choice="auto",
        truncation=request.truncation or "disabled",
        parallel_tool_calls=True,
        text_format=text_format,
        top_p=request.top_p if request.top_p is not None else 1.0,
        temperature=request.temperature if request.temperature is not None else 1.0,
        reasoning=reasoning_out,
        max_output_tokens=request.max_output_tokens,
        completed_at=int(time.time()),
    )


# Streaming converter for Responses API
class ResponsesStreamConverter:
    """Convert NVIDIA OpenAI SSE stream to Responses API SSE events.

    Event sequence (per OpenAI Responses API streaming docs):
    - response.created
    - response.in_progress
    - response.output_item.added (for reasoning, function_call, or message)
    - response.reasoning_summary_text.delta (for reasoning)
    - response.reasoning_summary_text.done
    - response.output_item.done (for reasoning)
    - response.function_call_arguments.delta (for function calls)
    - response.function_call_arguments.done
    - response.output_item.done (for function_call)
    - response.content_part.added (for message text)
    - response.output_text.delta (for message text)
    - response.output_text.done
    - response.output_item.done (for message)
    - response.completed
    """

    def __init__(self, response_id: str, item_id: str, model: str, request: ResponsesRequest):
        self.response_id = response_id
        self.item_id = item_id
        self.model = model
        self.request = request
        self.first_write = True
        self.output_index = 0
        self.content_index = 0
        self.content_started = False
        self.tool_calls_sent = False
        self.accumulated_text = ""
        self.sequence_number = 0

        # Reasoning state
        self.accumulated_thinking = ""
        self.reasoning_item_id = ""
        self.reasoning_started = False
        self.reasoning_done = False

        # Tool calls state
        self.tool_call_items: list[dict] = []

    def _new_event(self, event_type: str, data: dict[str, Any]) -> bytes:
        data["type"] = event_type
        data["sequence_number"] = self.sequence_number
        self.sequence_number += 1
        return _sse_event(event_type, data)

    def _build_response_object(self, status: str, output: list[Any], usage: dict[str, Any] | None) -> dict[str, Any]:
        """Build a full response object for streaming events."""
        instructions = self.request.instructions if self.request.instructions else None

        truncation = "disabled"
        if self.request.truncation:
            truncation = self.request.truncation

        tools = []
        for t in self.request.tools:
            tools.append({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "strict": t.strict,
                "parameters": t.parameters,
            })
        if not tools:
            tools = []

        text_format = {"type": "text"}
        if self.request.text and self.request.text.format:
            text_format = {"type": self.request.text.format.type}
            if self.request.text.format.name:
                text_format["name"] = self.request.text.format.name
            if self.request.text.format.schema:
                text_format["schema"] = self.request.text.format.schema
            if self.request.text.format.strict is not None:
                text_format["strict"] = self.request.text.format.strict

        reasoning = None
        if self.request.reasoning.effort or self.request.reasoning.summary:
            reasoning = {}
            if self.request.reasoning.effort:
                reasoning["effort"] = self.request.reasoning.effort
            if self.request.reasoning.summary:
                reasoning["summary"] = self.request.reasoning.summary

        top_p = 1.0
        if self.request.top_p is not None:
            top_p = self.request.top_p

        temperature = 1.0
        if self.request.temperature is not None:
            temperature = self.request.temperature

        return {
            "id": self.response_id,
            "object": "response",
            "created_at": int(time.time()),
            "completed_at": None,
            "status": status,
            "incomplete_details": None,
            "model": self.model,
            "previous_response_id": None,
            "instructions": instructions,
            "output": output,
            "error": None,
            "tools": tools,
            "tool_choice": "auto",
            "truncation": truncation,
            "parallel_tool_calls": True,
            "text": {"format": text_format},
            "top_p": top_p,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "top_logprobs": 0,
            "temperature": temperature,
            "reasoning": reasoning,
            "usage": usage,
            "max_output_tokens": self.request.max_output_tokens,
            "max_tool_calls": None,
            "store": False,
            "background": self.request.background,
            "service_tier": "default",
            "metadata": {},
            "safety_identifier": None,
            "prompt_cache_key": None,
        }

    def _create_response_created_event(self) -> bytes:
        return self._new_event("response.created", {
            "response": self._build_response_object("in_progress", [], None),
        })

    def _create_response_in_progress_event(self) -> bytes:
        return self._new_event("response.in_progress", {
            "response": self._build_response_object("in_progress", [], None),
        })

    def process(self, openai_chunk: dict[str, Any]) -> list[bytes]:
        """Process an OpenAI SSE chunk and return Responses API events."""
        events: list[bytes] = []

        choices = openai_chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return events

        choice = choices[0]
        if not isinstance(choice, dict):
            return events

        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}

        has_tool_calls = "tool_calls" in delta and isinstance(delta["tool_calls"], list) and delta["tool_calls"]
        has_thinking = "thinking" in delta and delta["thinking"]
        has_content = "content" in delta and delta["content"]

        # First chunk - emit initial events
        if self.first_write:
            self.first_write = False
            events.append(self._create_response_created_event())
            events.append(self._create_response_in_progress_event())

        # Handle reasoning/thinking (emitted first)
        if has_thinking:
            events.extend(self._process_thinking(delta["thinking"]))

        # Handle tool calls
        if has_tool_calls:
            events.extend(self._process_tool_calls(delta["tool_calls"]))
            self.tool_calls_sent = True

        # Handle text content (only if no tool calls)
        if not has_tool_calls and not self.tool_calls_sent and has_content:
            events.extend(self._process_text_content(delta["content"]))

        # Handle completion
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            events.extend(self._process_completion(finish_reason, openai_chunk))

        return events

    def _process_thinking(self, thinking: str) -> list[bytes]:
        events: list[bytes] = []

        # Start reasoning item if not started
        if not self.reasoning_started:
            self.reasoning_started = True
            self.reasoning_item_id = f"rs_{random.randint(100000, 999999)}"

            events.append(self._new_event("response.output_item.added", {
                "output_index": self.output_index,
                "item": {
                    "id": self.reasoning_item_id,
                    "type": "reasoning",
                    "summary": [],
                },
            }))

        # Accumulate thinking
        self.accumulated_thinking += thinking

        # Emit delta
        events.append(self._new_event("response.reasoning_summary_text.delta", {
            "item_id": self.reasoning_item_id,
            "output_index": self.output_index,
            "summary_index": 0,
            "delta": thinking,
        }))

        return events

    def _finish_reasoning(self) -> list[bytes]:
        if not self.reasoning_started or self.reasoning_done:
            return []
        self.reasoning_done = True

        events = [
            self._new_event("response.reasoning_summary_text.done", {
                "item_id": self.reasoning_item_id,
                "output_index": self.output_index,
                "summary_index": 0,
                "text": self.accumulated_thinking,
            }),
            self._new_event("response.output_item.done", {
                "output_index": self.output_index,
                "item": {
                    "id": self.reasoning_item_id,
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": self.accumulated_thinking}],
                    "encrypted_content": self.accumulated_thinking,
                },
            }),
        ]
        self.output_index += 1
        return events

    def _process_tool_calls(self, tool_calls: list[dict]) -> list[bytes]:
        events: list[bytes] = []

        # Finish reasoning first if it was started
        events.extend(self._finish_reasoning())

        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            oai_index = int(tc.get("index") or 0)

            fc_item_id = f"fc_{random.randint(100000, 999999)}_{i}"

            # Store for final output
            function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            tool_call_item = {
                "id": fc_item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": str(tc.get("id") or f"call_{oai_index}"),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or ""),
            }
            self.tool_call_items.append(tool_call_item)

            # response.output_item.added for function call
            events.append(self._new_event("response.output_item.added", {
                "output_index": self.output_index + i,
                "item": {
                    "id": fc_item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": tool_call_item["call_id"],
                    "name": tool_call_item["name"],
                    "arguments": "",
                },
            }))

            # response.function_call_arguments.delta
            args = function.get("arguments", "")
            if args:
                events.append(self._new_event("response.function_call_arguments.delta", {
                    "item_id": fc_item_id,
                    "output_index": self.output_index + i,
                    "delta": args,
                }))

            # response.function_call_arguments.done
            events.append(self._new_event("response.function_call_arguments.done", {
                "item_id": fc_item_id,
                "output_index": self.output_index + i,
                "arguments": args,
            }))

            # response.output_item.done for function call
            events.append(self._new_event("response.output_item.done", {
                "output_index": self.output_index + i,
                "item": tool_call_item,
            }))

        return events

    def _process_text_content(self, content: str) -> list[bytes]:
        events: list[bytes] = []

        # Finish reasoning first if it was started
        events.extend(self._finish_reasoning())

        # Emit output item and content part for first text content
        if not self.content_started:
            self.content_started = True

            # response.output_item.added
            events.append(self._new_event("response.output_item.added", {
                "output_index": self.output_index,
                "item": {
                    "id": self.item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }))

            # response.content_part.added
            events.append(self._new_event("response.content_part.added", {
                "item_id": self.item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "part": {"type": "output_text", "text": ""},
            }))

        # Accumulate and emit delta
        self.accumulated_text += content
        events.append(self._new_event("response.output_text.delta", {
            "item_id": self.item_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "delta": content,
        }))

        return events

    def _process_completion(self, finish_reason: str, openai_chunk: dict[str, Any]) -> list[bytes]:
        events: list[bytes] = []

        # Finish any pending reasoning
        events.extend(self._finish_reasoning())

        # Finish text content if started
        if self.content_started:
            events.append(self._new_event("response.output_text.done", {
                "item_id": self.item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "text": self.accumulated_text,
            }))
            events.append(self._new_event("response.output_item.done", {
                "output_index": self.output_index,
                "item": {
                    "id": self.item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": self.accumulated_text, "annotations": []}],
                },
            }))
            self.output_index += 1

        # Build usage
        usage_data = openai_chunk.get("usage")
        usage = None
        if isinstance(usage_data, dict):
            usage = {
                "input_tokens": int(usage_data.get("prompt_tokens") or 0),
                "output_tokens": int(usage_data.get("completion_tokens") or 0),
                "total_tokens": int(usage_data.get("total_tokens") or 0),
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }

        # response.completed
        status = "completed"
        if finish_reason == "length":
            status = "incomplete"

        events.append(self._new_event("response.completed", {
            "response": self._build_response_object(status, [], usage),
        }))

        return events

