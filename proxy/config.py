"""Configuration constants for the Atlas proxy — all env-overridable.

Logging is initialised in ``logger.py`` on import.  Use
``from .logger import get_logger, log`` in other modules instead of
``logging.getLogger()``.

Quick reference (env vars):
    Default model (OR)          ATLAS_OPENROUTER_MODEL
    Default model (HF)          ATLAS_HF_MODEL
    Force default (global)      FORCE_DEFAULT_MODEL=0|1
    Force default (per)         FORCE_DEFAULT_MODEL_OR / _HF
    Listen host / port          LISTEN_HOST / LISTEN_PORT
    Key file overrides          ATLAS_OPENROUTER_KEYS_FILE / ATLAS_HF_KEYS_FILE
    Runtime config file         data/proxy_data/runtime_provider.json
    System prompt override      data/proxy_data/prompt_override.txt
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .logger import get_logger, log


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------
def _env(key: str, default: str) -> str:
    """Read a string env var, falling back to *default*."""
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean env var (1/true/yes/on → True)."""
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
# Paths
# ---------------------------------------------------------------------------
_root_dir: Path = Path(__file__).resolve().parent.parent
DATA_DIR: str = str(_root_dir / "data")

# Runtime files written by the Atlas CLI (``atlas restart --huggingface``)
RUNTIME_DIR: str = str(_root_dir / "data" / "proxy_data")
RUNTIME_PROVIDER_FILE: str = str(_root_dir / "data" / "proxy_data" / "runtime_provider.json")

# System prompt override hot-reload target
SYSTEM_PROMPT_OVERRIDE_FILE: str = _env(
    "SYSTEM_PROMPT_OVERRIDE_FILE",
    str(_root_dir / "data" / "proxy_data" / "prompt_override.txt"),
)

# Key file locations (paths verified by tests)
KEY_FILE: str = _env(
    "ATLAS_OPENROUTER_KEYS_FILE",
    _env("KEY_FILE", str(_root_dir / "data" / "openrouter_data" / "openroute_keys.txt")),
)
FALLBACK_KEY_FILE: str = _env(
    "FALLBACK_KEY_FILE",
    str(_root_dir / "data" / "openrouter_data" / "openroute_keys.txt"),
)
HF_KEY_FILE: str = _env("ATLAS_HF_KEYS_FILE", str(_root_dir / "data" / "huggingface_data" / "hf_keys.txt"))
HF_DEAD_KEYS_FILE: str = _env(
    "ATLAS_HF_DEAD_KEYS_FILE",
    str(_root_dir / "data" / "huggingface_data" / "dead_hf_keys.txt"),
)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
#
# The Provider class and the registry live in `proxy.providers`. This block
# just registers the two built-in providers (OpenRouter, HuggingFace) and
# keeps the historical `OPENROUTER_CONFIG` / `HF_CONFIG` / `PROVIDERS` names
# available as back-compat shims. New code should import Provider, etc.,
# directly from `proxy.providers`.
from .providers import (
    Provider,
    ProviderCapability,
    PoolMode,
    register_provider,
    get_provider,
    get_active_provider,
    list_providers,
    provider_labels,
    resolve_provider_name,
    set_active_provider_name as _set_active_provider_name,
    _is_hf_quota as _provider_is_hf_quota,
    _is_hf_key_dead as _provider_is_hf_key_dead,
)

# Register the built-in providers. Tests or deployments can call
# ``register_provider(...)`` with allow_override=True to swap either out
# before any key pools are built.
from .providers import _build_openrouter, _build_huggingface  # noqa: E402

OPENROUTER_CONFIG: Provider = _build_openrouter()
# Backfill key file paths from the env vars loaded above.
OPENROUTER_CONFIG = Provider(
    **{**OPENROUTER_CONFIG.__dict__,
       "key_file": KEY_FILE,
       "fallback_key_file": FALLBACK_KEY_FILE}
)
register_provider(OPENROUTER_CONFIG, allow_override=True)

HF_CONFIG: Provider = _build_huggingface()
HF_CONFIG = Provider(
    **{**HF_CONFIG.__dict__,
       "key_file": HF_KEY_FILE,
       "dead_keys_file": HF_DEAD_KEYS_FILE}
)
register_provider(HF_CONFIG, allow_override=True)

PROVIDERS: dict[str, Provider] = {
    "openrouter": OPENROUTER_CONFIG,
    "huggingface": HF_CONFIG,
}


# ---------------------------------------------------------------------------
# Provider selection — runtime config file written by the Atlas CLI
# ---------------------------------------------------------------------------
# atlas restart                     → OpenRouter (default)
# atlas restart --huggingface       → Hugging Face
# atlas restart --huggingface --M   → Hugging Face (M)
def _load_runtime_provider() -> str:
    """Read provider from runtime config file (written by the Atlas CLI).

    Returns ``openrouter`` or ``huggingface``.
    Falls back to ``ATLAS_PROVIDER`` env var, then ``openrouter``.

    The runtime provider file controls which upstream's keys are loaded and
    which API surface is exposed, so it is a security-sensitive control. We
    refuse to load it if it is world-writable (any local user can switch
    the proxy's upstream by editing one JSON line).
    """
    try:
        import json as _json
        import os as _os
        import stat as _stat
        from pathlib import Path as _Path
        p = _Path(RUNTIME_PROVIDER_FILE)
        if p.is_file():
            # Check ownership: refuse to load if world-writable.
            try:
                st = p.stat()
                if st.st_mode & _stat.S_IWOTH:
                    log.warning(
                        "Refusing to load %s: file is world-writable (uid=%d mode=%04o). "
                        "Fix with: chmod 644 %s",
                        RUNTIME_PROVIDER_FILE, st.st_uid, st.st_mode & 0o777, RUNTIME_PROVIDER_FILE,
                    )
                    return _env("ATLAS_PROVIDER", "openrouter")
            except OSError:
                pass  # stat failure handled below
            data = _json.loads(p.read_text(encoding="utf-8"))
            provider = data.get("provider", "").lower()
            if provider in ("huggingface", "hf", "openrouter", "or"):
                return "huggingface" if provider in ("huggingface", "hf") else "openrouter"
    except Exception as e:
        log.warning("Failed to load runtime provider from %s: %s", RUNTIME_PROVIDER_FILE, e)
    return _env("ATLAS_PROVIDER", "openrouter")


def _load_runtime_model() -> Optional[str]:
    """Read model override from runtime config file (any provider)."""
    try:
        import json as _json
        from pathlib import Path as _Path
        p = _Path(RUNTIME_PROVIDER_FILE)
        if p.is_file():
            data = _json.loads(p.read_text(encoding="utf-8"))
            model = data.get("model", "").strip()
            if model:
                return model
    except Exception:
        pass
    return None


# Backwards-compatible base URL constants. New code should use
# ``get_active_provider().chat_url`` / ``messages_url`` / ``models_url``.
OPENROUTER_BASE_URL: str = OPENROUTER_CONFIG.base_url
OPENROUTER_CHAT: str = OPENROUTER_CONFIG.chat_url
OPENROUTER_MESSAGES: str = OPENROUTER_CONFIG.messages_url
OPENROUTER_MODELS: str = OPENROUTER_CONFIG.models_url
HF_BASE_URL: str = HF_CONFIG.base_url
HF_KEY_PREFIX: str = HF_CONFIG.key_prefix
HF_DEFAULT_MODEL: str = HF_CONFIG.default_model

# Active provider: read at import time from the runtime config file written
# by the Atlas CLI. ``proxy.providers.set_active_provider_name`` makes this
# available to any code that asks ``get_active_provider()``.
_set_active_provider_name(_load_runtime_provider())
PROVIDER: str = _load_runtime_provider()  # back-compat: legacy string name
"""Active provider canonical name: ``openrouter`` (default) or ``huggingface``.

Determined at import time from the runtime config file written by the
Atlas CLI.  ``get_active_provider()`` is the preferred accessor at call
sites -- it returns the full ``Provider`` record (URLs, capabilities,
auth scheme, error hooks) instead of just the name string.
"""


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
def _load_or_default_model() -> str:
    """Runtime model override (from runtime_provider.json) wins over env/env-default."""
    model = _load_runtime_model()
    if model:
        return model
    return _env("ATLAS_OPENROUTER_MODEL", _env("OPENROUTER_MODEL", "minimax/minimax-m3:free"))


OPENROUTER_MODEL: str = _load_or_default_model()
"""Default OpenRouter model injected when the client omits ``model``."""

HF_DEFAULT_MODEL: str = _env("ATLAS_HF_MODEL", "deepseek-ai/DeepSeek-V4-Flash:deepinfra")
"""Fallback HF model (used only when runtime config has no model)."""

HF_MODEL: str = _load_runtime_model() or HF_DEFAULT_MODEL
"""Resolved HF model: runtime config → env → built-in default."""


# ---------------------------------------------------------------------------
# Model override flags — per-provider, with backwards-compatible global
# ---------------------------------------------------------------------------
FORCE_DEFAULT_MODEL: bool = _env_bool("FORCE_DEFAULT_MODEL", True)
"""Global override. When True (default), the client's ``model`` is replaced
with the provider's configured default. When False, the client's model
passes through — same as setting both ``FORCE_DEFAULT_MODEL_OR`` and
``FORCE_DEFAULT_MODEL_HF`` to False."""

FORCE_DEFAULT_MODEL_OR: bool = _env_bool("FORCE_DEFAULT_MODEL_OR", FORCE_DEFAULT_MODEL)
"""When True, override the client-sent model with ``OPENROUTER_MODEL``.
Takes precedence over the global flag when this env var is explicitly set."""

FORCE_DEFAULT_MODEL_HF: bool = _env_bool("FORCE_DEFAULT_MODEL_HF", FORCE_DEFAULT_MODEL)
"""When True, override the client-sent model with ``HF_MODEL``.
Takes precedence over the global flag when this env var is explicitly set."""


def get_force_default_model() -> bool:
    """Return the force-override flag for the *active* provider.

    Single entry point for all override logic -- resolves the correct
    per-provider flag while preserving backwards compatibility for
    deployments that only set the global ``FORCE_DEFAULT_MODEL``.
    """
    if PROVIDER == "huggingface":
        return FORCE_DEFAULT_MODEL_HF
    return FORCE_DEFAULT_MODEL_OR


def get_provider() -> str:
    """Return the active provider's canonical name.

    Prefer ``get_active_provider()`` from ``proxy.providers`` if you
    want the full ``Provider`` record instead of just the name.
    """
    return PROVIDER


def get_provider_config(name: Optional[str] = None) -> Provider:
    """Return the ``Provider`` for *name* (defaults to the active one)."""
    if name is None:
        return get_active_provider()
    p = get_provider(name)
    return p if p is not None else get_active_provider()


def get_chat_url() -> str:
    """Upstream chat/completions URL for the active provider."""
    return get_active_provider().chat_url


def get_messages_url() -> str:
    """Upstream messages URL for the active provider."""
    return get_active_provider().messages_url


def get_models_url() -> str:
    """Upstream models URL for the active provider."""
    return get_active_provider().models_url


def get_default_model() -> str:
    """Default model for the active provider (with runtime override applied)."""
    override = _load_runtime_model()
    if override:
        return override
    return get_active_provider().default_model


def get_key_file() -> str:
    """Key file path for the active provider."""
    return get_active_provider().key_file


def get_fallback_key_file() -> str:
    """Fallback key file path (for OpenRouter when primary is empty)."""
    return FALLBACK_KEY_FILE


def get_key_prefix() -> str:
    """Key prefix for the active provider."""
    return get_active_provider().key_prefix


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
LISTEN_HOST: str = _env("LISTEN_HOST", "0.0.0.0")
# Forked proxy defaults to 8777 — the primary atlas proxy owns 8788.
# Override with LISTEN_PORT env var if you need a different port.
LISTEN_PORT: int = _env_int("LISTEN_PORT", 8777)


# ---------------------------------------------------------------------------
# Connection pool / timeouts
# ---------------------------------------------------------------------------
MAX_CONNECTIONS: int = _env_int("MAX_CONNECTIONS", 200)
MAX_KEEPALIVE_CONNECTIONS: int = _env_int("MAX_KEEPALIVE_CONNECTIONS", 100)
KEEPALIVE_EXPIRY: float = _env_float("KEEPALIVE_EXPIRY", 60.0)
CONNECT_TIMEOUT: float = _env_float("CONNECT_TIMEOUT", 15.0)
READ_TIMEOUT: float = _env_float("ATLAS_PROXY_READ_TIMEOUT", _env_float("READ_TIMEOUT", 600.0))
WRITE_TIMEOUT: float = _env_float("WRITE_TIMEOUT", 300.0)
POOL_TIMEOUT: float = _env_float("POOL_TIMEOUT", 30.0)


# ---------------------------------------------------------------------------
# Key health
# ---------------------------------------------------------------------------
COOLDOWN_BASE_SECONDS: float = _env_float(
    "ATLAS_PROXY_COOLDOWN_SECONDS", _env_float("COOLDOWN_BASE_SECONDS", 45.0)
)
COOLDOWN_MAX_SECONDS: float = _env_float("COOLDOWN_MAX_SECONDS", 300.0)
MAX_CONSECUTIVE_ERRORS: int = _env_int(
    "ATLAS_PROXY_MAX_ERRORS", _env_int("MAX_CONSECUTIVE_ERRORS", 8)
)
SUSPEND_SECONDS: float = _env_float(
    "ATLAS_PROXY_SUSPEND_SECONDS", _env_float("SUSPEND_SECONDS", 600.0)
)
HEALTH_CHECK_INTERVAL: float = _env_float("HEALTH_CHECK_INTERVAL", 60.0)
PREWARM_INTERVAL: float = _env_float("PREWARM_INTERVAL", 300.0)


# ---------------------------------------------------------------------------
# Retry / streaming
# ---------------------------------------------------------------------------
MAX_RETRIES: int = _env_int("ATLAS_PROXY_MAX_RETRIES", _env_int("MAX_RETRIES", 5))
# 409 (Conflict) intentionally excluded: it's not safe to auto-retry for
# most APIs (state conflicts, idempotency-key collisions, etc.) and
# retrying a non-idempotent op can produce duplicate side effects.
RETRY_STATUSES: frozenset[int] = frozenset(
    {408, 423, 425, 429, 499, 500, 502, 503, 504, 507, 524, 529}
)
STREAM_FIRST_BYTE_TIMEOUT: float = _env_float("STREAM_FIRST_BYTE_TIMEOUT", 20.0)
PROXY_KEEPALIVE_SECONDS: float = _env_float("ATLAS_PROXY_KEEPALIVE_SECONDS", 15.0)

# Max request body size in bytes. 10 MiB matches typical Anthropic/OpenAI
# request caps. Set to 0 to disable. The guard rejects before the body is
# buffered in memory (cheap to check from Content-Length; the streaming
# case is caught by a hard cap on the read).
MAX_REQUEST_BODY_BYTES: int = _env_int("ATLAS_PROXY_MAX_REQUEST_BODY_BYTES", 10 * 1024 * 1024)

# Max concurrent streams toward free-tier models (Nvidia worker limit is ~32).
# Stay under this to avoid mid-stream ResourceExhausted.
FREE_MODEL_MAX_CONCURRENT: int = _env_int("FREE_MODEL_MAX_CONCURRENT", 25)

# Per-key concurrency cap for partial-sticky load balancing. Once a key's
# in-flight count reaches this, ``next_key()`` rotates to the next healthy key.
MAX_CONCURRENT_PER_KEY: int = _env_int(
    "ATLAS_PROXY_MAX_CONCURRENT_PER_KEY", _env_int("MAX_CONCURRENT_PER_KEY", 24)
)

# Max consecutive successful requests a partial-sticky (OpenRouter) pool will
# serve from one key before proactively rotating. Set to 0 to disable
# (classic infinite stickiness).
STICKY_MAX_USES: int = _env_int("ATLAS_PROXY_STICKY_MAX_USES", 18)


# ---------------------------------------------------------------------------
# Context window safety — proactive truncation before upstream 400s
# ---------------------------------------------------------------------------
MAX_INPUT_TOKENS: int = _env_int(
    "ATLAS_PROXY_MAX_INPUT_TOKENS", _env_int("MAX_INPUT_TOKENS", 180_000)
)
"""Hard ceiling on estimated input tokens. When the accumulated message list
exceeds this, oldest messages are trimmed before forwarding to upstream.
Prevents 400 context-length errors and the /compact retry loop when Claude
Code (1M token assumption) exceeds the backing model's ~200K window.

Set to 0 to disable. Default 180K leaves ~20K headroom for a 200K model."""

MAX_TOKEN_TRIM_KEEP_SYSTEM: bool = _env_bool("MAX_TOKEN_TRIM_KEEP_SYSTEM", True)
"""When trimming messages, do not remove system/developer messages."""

MODEL_CONTEXT_WINDOW: int = _env_int(
    "ATLAS_MODEL_CONTEXT_WINDOW", _env_int("MODEL_CONTEXT_WINDOW", 200_000)
)
"""Reported context window for the backing model. Advertised via /v1/models
so Claude Code triggers /compact at the right threshold (~85%)."""


# ---------------------------------------------------------------------------
# Non-streaming response size cap
# ---------------------------------------------------------------------------
MAX_RESPONSE_BYTES: int = _env_int("ATLAS_MAX_RESPONSE_BYTES", _env_int("MAX_RESPONSE_BYTES", 50 * 1024 * 1024))
"""Maximum response body size (bytes). 0 = unlimited. When set, non-streaming
responses exceeding this size trigger a 413 instead of full buffering.
Set to ~50MB to cap worst-case memory under MAX_CONNECTIONS=200."""


# ---------------------------------------------------------------------------
# Response API state passthrough
# ---------------------------------------------------------------------------
# previous_response_id, conversation, store, etc. are stripped because the
# proxy maps Responses API → chat/completions (stateless), and the upstream
# chat/completions endpoint doesn't support server-side conversation state.
# Clients relying on server-side continuity must send full history each request.


# ---------------------------------------------------------------------------
# System prompt override (loaded here after logger init)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_OVERRIDE_ENABLED: bool = _env_bool("SYSTEM_PROMPT_OVERRIDE_ENABLED", True)
SYSTEM_PROMPT_OVERRIDE: str = ""
SYSTEM_PROMPT_REINFORCEMENT_ENABLED: bool = _env_bool("SYSTEM_PROMPT_REINFORCEMENT_ENABLED", False)
SYSTEM_PROMPT_STRIP_HARNESS: bool = _env_bool("SYSTEM_PROMPT_STRIP_HARNESS", True)


def _load_system_prompt_override() -> str:
    """Read the override file from disk. Returns '' if missing/disabled."""
    if not SYSTEM_PROMPT_OVERRIDE_ENABLED:
        return ""
    try:
        p = Path(SYSTEM_PROMPT_OVERRIDE_FILE)
        if p.is_file():
            text = p.read_text(encoding="utf-8").strip()
            return text or ""
    except Exception as e:
        log.warning("Failed to load system prompt override from %s: %s", SYSTEM_PROMPT_OVERRIDE_FILE, e)
    return ""


SYSTEM_PROMPT_OVERRIDE = _load_system_prompt_override()
if SYSTEM_PROMPT_OVERRIDE:
    log.info("Loaded system prompt override (%d chars)", len(SYSTEM_PROMPT_OVERRIDE))
else:
    log.info("System prompt override disabled or file not found at %s", SYSTEM_PROMPT_OVERRIDE_FILE)


def reload_system_prompt_override() -> str:
    """Hot-reload the system prompt override file.

    Returns the new content (may be ''). Call from the periodic reload loop
    in ``main.py`` so editing the file takes effect without a restart.
    """
    new_text = _load_system_prompt_override()
    old_len = len(SYSTEM_PROMPT_OVERRIDE)
    if new_text != SYSTEM_PROMPT_OVERRIDE:
        import proxy.config as _cfg
        _cfg.SYSTEM_PROMPT_OVERRIDE = new_text
        # Also update the imported reference in system_prompt.py
        import proxy.system_prompt as _sp
        _sp.SYSTEM_PROMPT_OVERRIDE = new_text
        _sp.SYSTEM_PROMPT_REINFORCEMENT_ENABLED = _cfg.SYSTEM_PROMPT_REINFORCEMENT_ENABLED
        log.info(
            "System prompt override hot-reloaded (%d → %d chars)",
            old_len,
            len(new_text),
        )
    return new_text


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
CORS_ORIGINS: List[str] = [
    o.strip() for o in _env("CORS_ORIGINS", "").split(",") if o.strip()
]
"""Explicit CORS origin allowlist. Empty list = no CORS headers at all
(safest for server-to-server traffic). Wildcard "*" is rejected by
browsers when ``allow_credentials=True`` (see main.py); use explicit
origins like "https://app.example.com" for browser clients."""


# ---------------------------------------------------------------------------
# Debug / upstream identity
# ---------------------------------------------------------------------------
UPSTREAM_REFERER: str = _env("ATLAS_UPSTREAM_REFERER", "https://localhost:8777")
UPSTREAM_TITLE: str = _env("ATLAS_UPSTREAM_TITLE", "Atlas-Translation-Proxy")
SAVE_PAYLOAD_FILES: bool = _env_bool("ATLAS_SAVE_PAYLOAD_FILES", False)
"""If True, write incoming request payloads to PAYLOAD_DIR for debugging."""
PAYLOAD_DIR: str = _env("ATLAS_PAYLOAD_DIR", str(_root_dir / "data" / "payloads"))


# ---------------------------------------------------------------------------
# Logging (configured in logger.py on import)
# ---------------------------------------------------------------------------
LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
LOG_JSON: bool = _env_bool("LOG_JSON", False)
LOG_REQUEST_ID: bool = _env_bool("LOG_REQUEST_ID", True)


# ---------------------------------------------------------------------------
# Key lifecycle management (Hugging Face)
# ---------------------------------------------------------------------------
def load_dead_hf_keys() -> set:
    """Load retired HF keys from dead_hf_keys.txt.

    Returns an empty set if the file doesn't exist. Never raises.
    """
    try:
        p = Path(HF_DEAD_KEYS_FILE)
        if not p.is_file():
            return set()
        keys: set[str] = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.startswith("hf_"):
                keys.add(line)
        return keys
    except Exception as e:
        log.warning("Failed to load dead HF keys from %s: %s", HF_DEAD_KEYS_FILE, e)
        return set()


def filter_hf_keys(keys: list[str], dead_keys: set[str]) -> list[str]:
    """Filter out dead HF keys from the active key list.

    Preserves order. If the current sticky key is still alive, it stays first.
    Dead keys that appear in both files remain dead.
    """
    dead = dead_keys or set()
    return [k for k in keys if k not in dead]


def add_dead_hf_key(key: str) -> bool:
    """Append a key to dead_hf_keys.txt.

    Returns True if the key was added (not already present). Creates the
    file lazily. Never raises.
    """
    try:
        p = Path(HF_DEAD_KEYS_FILE)
        existing = load_dead_hf_keys()
        if key in existing:
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{key}\n")
        log.info("HF key added to dead_hf_keys.txt (will be removed from active file on reload)")
        return True
    except Exception as e:
        log.error("Failed to persist dead HF key: %s", e)
        return False


def remove_hf_key(key: str) -> bool:
    """Remove a key from hf_keys.txt.

    This is the counterpart of ``add_dead_hf_key()``: we add to
    dead_keys.txt AND remove from hf_keys.txt so the active file stays clean.

    Returns True if the key was present and removed. Never raises.
    """
    try:
        p = Path(HF_KEY_FILE)
        if not p.is_file():
            return False
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if line.strip() != key]
        if len(kept) == len(lines):
            return False  # key not in file
        p.write_text("".join(kept), encoding="utf-8")
        log.info("HF key removed from hf_keys.txt (%d lines → %d)", len(lines), len(kept))
        return True
    except Exception as e:
        log.error("Failed to remove HF key from hf_keys.txt: %s", e)
        return False


def retire_and_remove_hf_key(key: str) -> tuple[bool, bool]:
    """Convenience: add to dead_hf_keys.txt AND remove from hf_keys.txt.

    Returns (added_to_dead, removed_from_active). Both True means a clean
    two-sided retirement. Only the dead-file side True means the key was
    already gone from the active file (idempotent). Never raises.
    """
    added = add_dead_hf_key(key)
    removed = remove_hf_key(key)
    if added or removed:
        log.info("HF key retired: dead_file=%s active_file=%s", added, removed)
    return (added, removed)


def migrate_hf_active_keys() -> tuple[int, int]:
    """Startup migration: remove keys from hf_keys.txt that are already
    in dead_hf_keys.txt.

    Cleans up the inconsistent state caused by the old retirement path that
    only added to dead_keys.txt without removing from hf_keys.txt.

    Returns (removed_count, kept_count). Never raises.
    """
    try:
        active_p = Path(HF_KEY_FILE)
        if not active_p.is_file():
            return (0, 0)
        dead_keys = load_dead_hf_keys()
        if not dead_keys:
            return (0, 0)
        lines = active_p.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [
            line for line in lines
            if not (line.strip().startswith("hf_") and line.strip() in dead_keys)
        ]
        removed = len(lines) - len(kept)
        if removed > 0:
            active_p.write_text("".join(kept), encoding="utf-8")
            log.info(
                "Startup migration: removed %d dead keys from hf_keys.txt "
                "(kept %d usable keys)",
                removed, len(kept),
            )
        return (removed, len(kept))
    except Exception as e:
        log.warning("Failed to migrate HF keys at startup: %s", e)
        return (0, 0)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
# The provider-specific hooks (HF quota body markers, HF dead-key body
# markers) live in `proxy.providers`. These thin shims keep the old
# `is_hf_rate_limit_error` / `is_hf_key_invalid` names available for
# callers that imported them from `proxy.config` historically. New code
# should call `provider.check_quota(...)` / `provider.check_key_dead(...)`
# on the active provider.
def is_hf_rate_limit_error(status: int, body: Optional[bytes] = None) -> bool:
    """Back-compat shim -- delegates to ``HF_CONFIG.check_quota``."""
    return HF_CONFIG.check_quota(status, body)


def is_hf_key_invalid(status: int, body: Optional[bytes] = None) -> bool:
    """Back-compat shim -- delegates to ``HF_CONFIG.check_key_dead``."""
    return HF_CONFIG.check_key_dead(status, body)


# ---------------------------------------------------------------------------
# Backwards-compat aliases -- proxy.py historically imported HF-specific names
# ---------------------------------------------------------------------------
is_rate_limit_error = is_hf_rate_limit_error
is_key_invalid = is_hf_key_invalid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Env helpers
    "_env", "_env_bool", "_env_float", "_env_int",
    # Paths & data dirs
    "DATA_DIR", "RUNTIME_DIR", "RUNTIME_PROVIDER_FILE",
    "SYSTEM_PROMPT_OVERRIDE_FILE",
    "KEY_FILE", "FALLBACK_KEY_FILE",
    "HF_KEY_FILE", "HF_DEAD_KEYS_FILE",
    # Provider registry
    "ProviderConfig", "PROVIDERS",
    "PROVIDER", "get_provider", "get_provider_config",
    "OPENROUTER_BASE_URL", "OPENROUTER_CHAT", "OPENROUTER_MESSAGES",
    "OPENROUTER_MODELS",
    "OPENROUTER_MODEL", "OPENROUTER_CONFIG",
    "HF_BASE_URL", "HF_DEFAULT_MODEL", "HF_MODEL", "HF_KEY_PREFIX", "HF_CONFIG",
    # Model override
    "FORCE_DEFAULT_MODEL", "FORCE_DEFAULT_MODEL_OR", "FORCE_DEFAULT_MODEL_HF",
    "get_force_default_model",
    # URL / model / key resolution
    "get_chat_url", "get_messages_url", "get_models_url",
    "get_default_model", "get_key_file", "get_fallback_key_file", "get_key_prefix",
    # Server
    "LISTEN_HOST", "LISTEN_PORT",
    # Connection pool / timeouts
    "MAX_CONNECTIONS", "MAX_KEEPALIVE_CONNECTIONS", "KEEPALIVE_EXPIRY",
    "CONNECT_TIMEOUT", "READ_TIMEOUT", "WRITE_TIMEOUT", "POOL_TIMEOUT",
    # Key health
    "COOLDOWN_BASE_SECONDS", "COOLDOWN_MAX_SECONDS", "MAX_CONSECUTIVE_ERRORS",
    "SUSPEND_SECONDS", "HEALTH_CHECK_INTERVAL", "PREWARM_INTERVAL",
    # Retry / streaming
    "MAX_RETRIES", "RETRY_STATUSES", "STREAM_FIRST_BYTE_TIMEOUT",
    "PROXY_KEEPALIVE_SECONDS", "FREE_MODEL_MAX_CONCURRENT",
    "MAX_CONCURRENT_PER_KEY", "STICKY_MAX_USES",
    # Request body size cap
    "MAX_REQUEST_BODY_BYTES",
    # Context window safety
        "MAX_INPUT_TOKENS", "MAX_TOKEN_TRIM_KEEP_SYSTEM", "MODEL_CONTEXT_WINDOW",
    # Response size cap
    "MAX_RESPONSE_BYTES",
    # System prompt override
    "SYSTEM_PROMPT_OVERRIDE_ENABLED", "SYSTEM_PROMPT_OVERRIDE",
    "SYSTEM_PROMPT_REINFORCEMENT_ENABLED", "SYSTEM_PROMPT_STRIP_HARNESS",
    "reload_system_prompt_override",
    # CORS
    "CORS_ORIGINS",
    # Debug / upstream
    "UPSTREAM_REFERER", "UPSTREAM_TITLE",
    "SAVE_PAYLOAD_FILES", "PAYLOAD_DIR",
    # Logging
    "LOG_LEVEL", "LOG_JSON", "LOG_REQUEST_ID", "get_logger", "log",
    # Key lifecycle management
    "load_dead_hf_keys", "filter_hf_keys", "add_dead_hf_key",
    "remove_hf_key", "retire_and_remove_hf_key", "migrate_hf_active_keys",
    # Error classification
    "is_hf_rate_limit_error", "is_hf_key_invalid",
    "is_rate_limit_error", "is_key_invalid",
]
