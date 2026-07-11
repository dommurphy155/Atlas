"""System prompt override — strips harness junk and injects the override persona."""

import re
from pathlib import Path

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


def replace_system_prompt(body: dict, provider: str = "openai") -> dict:
    """Strip harness context and inject override persona.

    Mutates body in-place and returns it.
    - Strips <system-reminder> blocks from user messages
    - Replaces system messages with override
    - Injects override at start of first user message for primacy
    """
    override = _read_override()
    if not override:
        return body

    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return body

    # Find first user message index
    first_user_idx = None
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            first_user_idx = i
            break

    # Strip system-reminder junk from ALL messages
    for msg in messages:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            msg["content"] = _strip_system_junk(msg["content"])

    # Replace/insert system message with override
    system_found = False
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            msg["content"] = override
            system_found = True
            break

    if not system_found:
        # Insert system message at start
        messages.insert(0, {"role": "system", "content": override})

    # Also prepend override to first user message for double primacy
    if first_user_idx is not None:
        user_msg = messages[first_user_idx]
        user_content = user_msg.get("content", "")
        user_msg["content"] = f"{override}\n\n{user_content}"

    return body
