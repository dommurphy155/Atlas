"""Configuration constants — all env-overridable.

Logging is initialized in logger.py on import. Use `from .logger import get_logger, log`
in other modules instead of `logging.getLogger()`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .logger import get_logger, log, set_request_id, clear_request_id, bind_request_id

# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v is not None else default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None else default


# ---------------------------------------------------------------------------
# Model / upstream
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL: str = _env("ATLAS_OPENROUTER_BASE_URL", _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
OPENROUTER_CHAT: str = f"{OPENROUTER_BASE_URL}/chat/completions"
OPENROUTER_MESSAGES: str = f"{OPENROUTER_BASE_URL}/messages"
OPENROUTER_MODELS: str = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_RESPONSES: str = f"{OPENROUTER_BASE_URL}/responses"

OPENROUTER_MODEL: str = _env("ATLAS_OPENROUTER_MODEL", _env("OPENROUTER_MODEL", "nemotron-3-ultra-550b-a55b:free"))
"""Default model injected when the client omits "model" (or when FORCE_DEFAULT_MODEL)."""

FORCE_DEFAULT_MODEL: bool = _env_bool("FORCE_DEFAULT_MODEL", True)
"""If True, always override the client-supplied model with OPENROUTER_MODEL."""

# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------
KEY_FILE: str = _env("ATLAS_OPENROUTER_KEYS_FILE", _env("KEY_FILE", "/root/openrouter/scripts/data/openroute_keys.txt"))
FALLBACK_KEY_FILE: str = _env("FALLBACK_KEY_FILE", "/tmp/fake_keys/openroute_keys.txt")

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
LISTEN_HOST: str = _env("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT: int = _env_int("LISTEN_PORT", 8788)

# ---------------------------------------------------------------------------
# Connection pool / timeouts
# ---------------------------------------------------------------------------
MAX_CONNECTIONS: int = _env_int("MAX_CONNECTIONS", 200)
MAX_KEEPALIVE_CONNECTIONS: int = _env_int("MAX_KEEPALIVE_CONNECTIONS", 100)
KEEPALIVE_EXPIRY: float = _env_float("KEEPALIVE_EXPIRY", 60.0)
CONNECT_TIMEOUT: float = _env_float("CONNECT_TIMEOUT", 15.0)
READ_TIMEOUT: float = _env_float("READ_TIMEOUT", 300.0)
WRITE_TIMEOUT: float = _env_float("WRITE_TIMEOUT", 30.0)
POOL_TIMEOUT: float = _env_float("POOL_TIMEOUT", 30.0)

# ---------------------------------------------------------------------------
# Key health
# ---------------------------------------------------------------------------
COOLDOWN_BASE_SECONDS: float = _env_float("ATLAS_PROXY_COOLDOWN_SECONDS", _env_float("COOLDOWN_BASE_SECONDS", 45.0))
COOLDOWN_MAX_SECONDS: float = _env_float("COOLDOWN_MAX_SECONDS", 300.0)
MAX_CONSECUTIVE_ERRORS: int = _env_int("ATLAS_PROXY_MAX_ERRORS", _env_int("MAX_CONSECUTIVE_ERRORS", 8))
SUSPEND_SECONDS: float = _env_float("ATLAS_PROXY_SUSPEND_SECONDS", _env_float("SUSPEND_SECONDS", 600.0))
HEALTH_CHECK_INTERVAL: float = _env_float("HEALTH_CHECK_INTERVAL", 60.0)
PREWARM_INTERVAL: float = _env_float("PREWARM_INTERVAL", 300.0)

# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------
MAX_RETRIES: int = _env_int("MAX_RETRIES", 5)
RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
STREAM_FIRST_BYTE_TIMEOUT: float = _env_float("STREAM_FIRST_BYTE_TIMEOUT", 20.0)

# ---------------------------------------------------------------------------
# Logging (configured in logger.py on import)
# ---------------------------------------------------------------------------
LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
LOG_JSON: bool = _env_bool("LOG_JSON", False)
LOG_REQUEST_ID: bool = _env_bool("LOG_REQUEST_ID", True)

# ---------------------------------------------------------------------------
# System prompt override (loaded here after logger init)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_OVERRIDE_ENABLED: bool = _env_bool("SYSTEM_PROMPT_OVERRIDE_ENABLED", True)
SYSTEM_PROMPT_OVERRIDE_FILE: str = _env("SYSTEM_PROMPT_OVERRIDE_FILE", "/root/claude/atlas/data/system_prompt_override.txt")
SYSTEM_PROMPT_OVERRIDE: str = ""

if SYSTEM_PROMPT_OVERRIDE_ENABLED and Path(SYSTEM_PROMPT_OVERRIDE_FILE).is_file():
    try:
        SYSTEM_PROMPT_OVERRIDE = Path(SYSTEM_PROMPT_OVERRIDE_FILE).read_text(encoding="utf-8").strip()
        if SYSTEM_PROMPT_OVERRIDE:
            log.info("Loaded system prompt override (%d chars)", len(SYSTEM_PROMPT_OVERRIDE))
    except Exception as e:
        log.warning("Failed to load system prompt override from %s: %s", SYSTEM_PROMPT_OVERRIDE_FILE, e)
else:
    log.info("System prompt override disabled or file not found at %s", SYSTEM_PROMPT_OVERRIDE_FILE)

# ---------------------------------------------------------------------------
# CORS (comma-separated origins, or * for all)
# ---------------------------------------------------------------------------
CORS_ORIGINS: List[str] = [o.strip() for o in _env("CORS_ORIGINS", "*").split(",") if o.strip()]

# Re-export logger helpers
__all__ = [
    "OPENROUTER_MODEL",
    "FORCE_DEFAULT_MODEL",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_CHAT",
    "OPENROUTER_MESSAGES",
    "OPENROUTER_MODELS",
    "OPENROUTER_RESPONSES",
    "KEY_FILE",
    "FALLBACK_KEY_FILE",
    "LISTEN_HOST",
    "LISTEN_PORT",
    "MAX_CONNECTIONS",
    "MAX_KEEPALIVE_CONNECTIONS",
    "KEEPALIVE_EXPIRY",
    "CONNECT_TIMEOUT",
    "READ_TIMEOUT",
    "WRITE_TIMEOUT",
    "POOL_TIMEOUT",
    "COOLDOWN_BASE_SECONDS",
    "COOLDOWN_MAX_SECONDS",
    "MAX_CONSECUTIVE_ERRORS",
    "SUSPEND_SECONDS",
    "HEALTH_CHECK_INTERVAL",
    "PREWARM_INTERVAL",
    "MAX_RETRIES",
    "RETRY_STATUSES",
    "STREAM_FIRST_BYTE_TIMEOUT",
    "LOG_LEVEL",
    "LOG_JSON",
    "LOG_REQUEST_ID",
    "SYSTEM_PROMPT_OVERRIDE_ENABLED",
    "SYSTEM_PROMPT_OVERRIDE_FILE",
    "SYSTEM_PROMPT_OVERRIDE",
    "CORS_ORIGINS",
    "get_logger",
    "log",
    "set_request_id",
    "clear_request_id",
    "bind_request_id",
]