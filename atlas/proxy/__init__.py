"""OpenRouter ↔ OpenAI ↔ Anthropic Translation Proxy."""

__version__ = "1.1.0"

from .system_prompt import (
    _inject_system_override_openai,
    _inject_system_override_anthropic,
)

__all__ = [
    "_inject_system_override_openai",
    "_inject_system_override_anthropic",
]
