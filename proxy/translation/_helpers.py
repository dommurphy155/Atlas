"""Small shared utilities used by both request and message converters.

These are leaf-level helpers (no other translation module is imported
by them), so it's safe for every other sub-module to import from here.
"""
from __future__ import annotations

from typing import Any, List, Optional

from ._constants import _ANTHROPIC_BLOCK_TYPES, _EFFORT_ALIASES, _EFFORT_VALUES
from ..config import get_logger

log = get_logger(__name__)


def stringify_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        import orjson
        return orjson.dumps(args).decode()
    except Exception:
        return str(args)


def normalize_effort(value: Any) -> Optional[str]:
    """Map any effort-like value to a canonical OpenRouter effort string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Numeric budgets sometimes arrive as effort; treat large numbers as high.
        if value <= 0:
            return "none"
        if value < 1024:
            return "low"
        if value < 8192:
            return "medium"
        if value < 32768:
            return "high"
        return "xhigh"
    s = str(value).strip().lower()
    if not s:
        return None
    if s in _EFFORT_VALUES:
        return s
    if s in _EFFORT_ALIASES:
        return _EFFORT_ALIASES[s]
    # Unknown effort string — log a warning so the silent fallback to "medium"
    # is visible, then clamp to a safe default rather than passing through
    # something OpenRouter may reject.
    log.warning("Unknown reasoning_effort value %r — defaulting to 'medium'", s)
    return "medium"


def extract_budget(obj: Any) -> Optional[int]:
    """Pull a positive integer token budget from assorted shapes."""
    if obj is None:
        return None
    if isinstance(obj, (int, float)) and obj > 0:
        return int(obj)
    if isinstance(obj, dict):
        for key in (
            "budget_tokens",
            "max_tokens",
            "thinking_budget",
            "reasoning_budget",
            "tokens",
            "budget",
        ):
            v = obj.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
    return None


def is_anthropic_content_list(content: Any) -> bool:
    if not isinstance(content, list) or not content:
        return False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in _ANTHROPIC_BLOCK_TYPES and btype != "text":
            return True
        # Anthropic image / tool_result without explicit type still detectable
        if "tool_use_id" in block or "input_schema" in block:
            return True
        if isinstance(block.get("source"), dict) and block["source"].get("type") in (
            "base64",
            "url",
        ):
            return True
    return False


def messages_look_anthropic(messages: List[Any]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if is_anthropic_content_list(content):
            return True
        # Anthropic never uses role "tool"; presence of tool_use blocks already caught.
    return False


def messages_look_openai_tools(messages: List[Any]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            return True
        if "tool_calls" in msg:
            return True
    return False


# Legacy underscore aliases for callers that imported these under their
# private names from the inlined translation.py.
_stringify_args = stringify_args
_normalize_effort = normalize_effort
_extract_budget = extract_budget
_is_anthropic_content_list = is_anthropic_content_list
_messages_look_anthropic = messages_look_anthropic
_messages_look_openai_tools = messages_look_openai_tools
