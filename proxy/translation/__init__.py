"""Wire-format translation between Anthropic /messages and OpenAI chat/completions.

This package is a thin re-export shim that preserves the historical
``proxy.translation`` import path after the original 1840-line monolith
was decomposed in Round 3. The actual implementation lives in the
sub-modules:

  * ``_constants``  -- module-level constants (effort aliases, etc.)
  * ``_helpers``    -- small pure utilities (stringify, normalize, detect)
  * ``_tools``      -- tool / tool_choice conversion
  * ``_messages``   -- bidirectional messages array conversion
  * ``_reasoning``  -- reasoning / thinking-block normalisation
  * ``_requests``   -- prepare_chat_body + prepare_messages_body
  * ``_responses``  -- non-streaming response shape conversion
  * ``_sse``        -- streaming SSE conversion

Every name that was previously a top-level symbol in
``proxy/translation.py`` is re-exported here so existing callers
(routes.py, the test suite) keep working without changes.
"""
from __future__ import annotations

from ._constants import (
    _ANTHROPIC_BLOCK_TYPES,
    _EFFORT_ALIASES,
    _EFFORT_VALUES,
    _SAFE_FORWARD,
    _STRIP_AFTER_NORMALIZE,
    _THINKING_TYPES,
)
from ._helpers import (
    _extract_budget,
    _is_anthropic_content_list,
    _messages_look_anthropic,
    _messages_look_openai_tools,
    _normalize_effort,
    _stringify_args,
    extract_budget,
    is_anthropic_content_list,
    messages_look_anthropic,
    messages_look_openai_tools,
    normalize_effort,
    stringify_args,
)
from ._messages import (
    anthropic_messages_to_openai,
    openai_messages_to_anthropic,
)
from ._reasoning import (
    _collect_reasoning_hints,
    _emit_reasoning_for_openai_chat,
    _emit_thinking_for_anthropic,
    collect_reasoning_hints,
    emit_reasoning_for_openai_chat,
    emit_thinking_for_anthropic,
)
from ._requests import (
    _apply_max_tokens_alias,
    _CHAR_PER_TOKEN,
    _enforce_input_limit,
    _estimate_tokens,
    _message_tokens,
    _PRIVATE_KEYS,
    _sanitize_body,
    _strip_internal_from_messages,
    _strip_private_keys,
    _trim_message_content,
    _trim_messages,
    prepare_chat_body,
    prepare_messages_body,
)
from ._responses import (
    _map_openai_finish_reason_to_anthropic,
    map_finish_reason_anthropic_to_openai,
    map_finish_reason_openai_to_anthropic,
    openai_response_to_anthropic,
    promote_reasoning_to_content_in_chat_response,
    translate_usage_anthropic_to_openai,
    translate_usage_openai_to_anthropic,
)
from ._sse import openai_sse_to_anthropic_sse
from ._tools import (
    anthropic_tools_to_openai,
    convert_tool_choice_anthropic_to_openai,
    convert_tool_choice_openai_to_anthropic,
    openai_tools_to_anthropic,
)

__all__ = [
    # constants
    "_ANTHROPIC_BLOCK_TYPES",
    "_EFFORT_ALIASES",
    "_EFFORT_VALUES",
    "_SAFE_FORWARD",
    "_STRIP_AFTER_NORMALIZE",
    "_THINKING_TYPES",
    # helpers (public + legacy underscore aliases)
    "_extract_budget",
    "_is_anthropic_content_list",
    "_messages_look_anthropic",
    "_messages_look_openai_tools",
    "_normalize_effort",
    "_stringify_args",
    "extract_budget",
    "messages_look_anthropic",
    "messages_look_openai_tools",
    "normalize_effort",
    "stringify_args",
    # messages
    "anthropic_messages_to_openai",
    "openai_messages_to_anthropic",
    # reasoning
    "_collect_reasoning_hints",
    "_emit_reasoning_for_openai_chat",
    "_emit_thinking_for_anthropic",
    "collect_reasoning_hints",
    "emit_reasoning_for_openai_chat",
    "emit_thinking_for_anthropic",
    # requests
    "_apply_max_tokens_alias",
    "_CHAR_PER_TOKEN",
    "_enforce_input_limit",
    "_estimate_tokens",
    "_message_tokens",
    "_PRIVATE_KEYS",
    "_sanitize_body",
    "_strip_internal_from_messages",
    "_strip_private_keys",
    "_trim_message_content",
    "_trim_messages",
    "prepare_chat_body",
    "prepare_messages_body",
    # responses
    "_map_openai_finish_reason_to_anthropic",
    "map_finish_reason_anthropic_to_openai",
    "map_finish_reason_openai_to_anthropic",
    "openai_response_to_anthropic",
    "promote_reasoning_to_content_in_chat_response",
    "translate_usage_anthropic_to_openai",
    "translate_usage_openai_to_anthropic",
    # sse
    "openai_sse_to_anthropic_sse",
    # tools
    "anthropic_tools_to_openai",
    "convert_tool_choice_anthropic_to_openai",
    "convert_tool_choice_openai_to_anthropic",
    "openai_tools_to_anthropic",
]
