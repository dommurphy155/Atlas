"""System prompt override — strips harness junk and injects the override persona.

Handles both wire shapes the proxy sees:
- OpenAI (/v1/chat/completions): string ``content``, optional ``role: system`` message.
- Anthropic (/v1/messages): top-level ``system`` field (string or list of text
  blocks), and ``content`` that may be a string OR a list of typed blocks
  (``{"type": "text", "text": ...}``, ``tool_use``, ``tool_result``, ...).

The old version assumed ``content`` was always a string, so on the Anthropic
path it (a) never stripped <system-reminder> blocks nested inside text parts,
(b) missed the top-level ``system`` field entirely — leaving the real system
prompt in primacy — and (c) stringified the user's block list into a Python
repr via an f-string, corrupting the message. This version fixes all three.
"""

import re
from pathlib import Path
from typing import Any

OVERRIDE_PATH = Path(__file__).resolve().parents[1] / "data" / "system_prompt_override.txt"

# Regex to strip system-reminder blocks (non-greedy match)
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def _read_override() -> str | None:
    """Read the override file. Returns None if empty or missing."""
    try:
        text = OVERRIDE_PATH.read_text(encoding="utf-8").strip()
        return text if text else None
    except FileNotFoundError:
        return None


def _strip_system_junk(content: str) -> str:
    """Remove <system-reminder> blocks and normalize whitespace."""
    content = _SYSTEM_REMINDER_RE.sub("", content)
    return content.strip()


def _content_to_text(content: Any) -> str:
    """Flatten a message's content (str or list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    # Fallback: don't stringify arbitrary objects into the payload.
    return ""


def _strip_junk_in_content(content: Any) -> Any:
    """Strip <system-reminder> blocks from content, preserving its shape.

    - str content: return cleaned str.
    - list content: clean each text block in place, leave other block types
      (tool_use, tool_result, image, ...) untouched.
    - anything else: return unchanged.
    """
    if isinstance(content, str):
        return _strip_system_junk(content)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] = _strip_system_junk(str(block.get("text") or ""))
        return content
    return content


def _prepend_to_user_content(content: Any, override: str) -> Any:
    """Prepend the override to a user message's content, preserving shape.

    - str content: ``f"{override}\n\n{content}"``.
    - list content: prepend a fresh text block so the existing typed blocks
      (images, tool_result, ...) survive intact. The override gets primacy as
      the first block.
    """
    if isinstance(content, list):
        return [{"type": "text", "text": override}, *content]
    if isinstance(content, str):
        return f"{override}\n\n{content}" if content else override
    # Unknown shape — wrap the override as a text block and keep the original.
    return [{"type": "text", "text": override}]


def replace_system_prompt(body: dict, provider: str = "openai") -> dict:
    """Strip harness context and inject override persona.

    Mutates body in-place and returns it.

    OpenAI path:
    - Strips <system-reminder> blocks from message content
    - Replaces any ``role: system`` message with the override
    - Prepends the override to the first user message for double primacy

    Anthropic path (provider="anthropic"):
    - Replaces the top-level ``system`` field (string or list of text blocks)
      with the override — this is where Claude Code's real system prompt lives
    - Strips <system-reminder> blocks from every message's text content,
      including text blocks nested inside block-list content
    - Prepends the override to the first user message (as a leading text block
      if content is a list) for double primacy
    """
    override = _read_override()
    if not override:
        return body

    messages = body.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    # ── Anthropic: override the top-level `system` field ──────────────
    # Claude Code sends its real system prompt here, NOT as a role:system
    # message. If we don't touch this, the original system prompt retains
    # primacy and the override loses. Represent it as a single text block so
    # anthropic_messages_to_openai() picks it up cleanly.
    if provider == "anthropic":
        body["system"] = [{"type": "text", "text": override}]

    # Find first user message index (before we mutate the list).
    first_user_idx = None
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            first_user_idx = i
            break

    # Strip system-reminder junk from ALL messages, preserving content shape.
    for msg in messages:
        if isinstance(msg, dict):
            msg["content"] = _strip_junk_in_content(msg.get("content"))

    # ── OpenAI: replace/insert a role:system message ────────────────────
    # On the Anthropic path there is no role:system message (the system prompt
    # lives in body["system"], handled above), so this loop is a no-op there.
    system_found = False
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            msg["content"] = override
            system_found = True
            break

    if not system_found and provider != "anthropic":
        messages.insert(0, {"role": "system", "content": override})
        # The insert shifted indices; recompute first_user_idx if it was set.
        if first_user_idx is not None:
            first_user_idx += 1

    # No "double primacy" prepend: the override already lives in body["system"]
    # (Anthropic) or a role:system message (OpenAI), which is the single
    # authoritative slot. Re-pasting it into the first user message just made
    # the same instruction appear 3x and cluttered the conversation. One place,
    # one override.
    return body
