"""Centralised upstream-provider registry.

This module is the single place where every piece of provider-specific
behaviour lives. To add a new provider (NVIDIA, Anthropic-direct, Groq,
Together, OpenAI-direct, anything), create a ``Provider`` instance and
register it -- no other file needs to change.

The pattern is capability-driven, not name-driven: call sites ask
``provider.has(ProviderCapability.MESSAGES_ENDPOINT)`` instead of
``if PROVIDER == "openrouter"``. That makes the registry composable --
a provider can advertise any subset of capabilities and the call
sites just keep working.

Public surface:

    Provider                        -- the immutable per-provider record
    ProviderCapability              -- IntFlag of advertised behaviours
    PoolMode                        -- sticky-mode enum
    get_provider(name=None)         -- resolve a Provider by name (or
                                        ``"openrouter"`` if unset)
    get_active_provider()           -- the active provider (from
                                        runtime config / env)
    register_provider(provider)     -- add a new provider (or override
                                        an existing one at startup)
    resolve_provider_name(token)    -- map an alias (``"or"``, ``"hf"``)
                                        to the canonical name
    list_providers()                -- names in registration order

Backwards compatibility:

    The historical names from ``proxy.config`` -- ``ProviderConfig``,
    ``OPENROUTER_CONFIG``, ``HF_CONFIG``, ``PROVIDERS`` dict,
    ``PROVIDER`` global, ``is_hf_rate_limit_error``, ``is_hf_key_invalid``
    -- are re-exported from ``proxy.config`` for any caller that
    imported them. New code should import from this module instead.
"""
from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, FrozenSet, List, Optional, Tuple

log = logging.getLogger("atlas_proxy.providers")


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class ProviderCapability(enum.IntFlag):
    """What this provider can do. A provider advertises any subset.

    CHAT_COMPLETIONS       POST /v1/chat/completions (OpenAI-style)
    MESSAGES_ENDPOINT      POST /v1/messages (Anthropic-style);
                           providers without this translate Anthropic
                           requests into OpenAI chat/completions
    MODELS_LIST            GET  /v1/models (provider-side catalogue)
    STREAMING_DONE_SENTINEL  OpenAI-style ``[DONE]`` frame; absence
                           implies Anthropic-style ``message_stop``
    REQUIRES_BEARER_AUTH   Standard ``Authorization: Bearer <key>``
                           header. (Almost every provider does; only
                           flag this if you need different auth.)
    SUPPORTS_TOOL_CALLS    Accepts OpenAI-style tool_calls / tool_choice
    HF_QUOTA_BODY_MARKERS  Provider-specific quota markers in 4xx
                           bodies; used by key retirement
    POOL_FULL_STICKY       One key per request until errored; most
                           providers want ``partial_sticky`` instead
    """
    NONE = 0
    CHAT_COMPLETIONS = enum.auto()
    MESSAGES_ENDPOINT = enum.auto()
    MODELS_LIST = enum.auto()
    STREAMING_DONE_SENTINEL = enum.auto()
    REQUIRES_BEARER_AUTH = enum.auto()
    SUPPORTS_TOOL_CALLS = enum.auto()
    HF_QUOTA_BODY_MARKERS = enum.auto()
    POOL_FULL_STICKY = enum.auto()

    # Convenience: a "fully featured OpenAI-compatible" provider has the
    # usual four capabilities and bearer auth.
    OPENAI_COMPAT = (
        CHAT_COMPLETIONS
        | MODELS_LIST
        | STREAMING_DONE_SENTINEL
        | REQUIRES_BEARER_AUTH
        | SUPPORTS_TOOL_CALLS
    )

    # Convenience: an "Anthropic-direct" provider has the messages
    # endpoint but no [DONE] sentinel.
    ANTHROPIC_NATIVE = (
        MESSAGES_ENDPOINT
        | MODELS_LIST
        | REQUIRES_BEARER_AUTH
        | SUPPORTS_TOOL_CALLS
    )


class PoolMode(str, enum.Enum):
    """Key pool stickiness strategy.

    FULL_STICKY    One key per request, kept across calls until it
                   errors. Used when retries on a different key are
                   expensive (HF quota exhaustion means an alternate
                   key may not help).
    PARTIAL_STICKY Round-robin across healthy keys; retries rotate.
    """
    FULL_STICKY = "full_sticky"
    PARTIAL_STICKY = "partial_sticky"


# ---------------------------------------------------------------------------
# The Provider record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provider:
    """Immutable per-provider record.

    Every field has a sensible default. To add a new provider, copy one
    of the concrete instances below, change the values, and
    ``register_provider()`` it. The provider's ``default_model`` is the
    string injected when a client omits ``model`` (or when the
    ``FORCE_DEFAULT_MODEL`` flag is True for this provider).
    """
    # Identity
    name: str                                          # canonical ("openrouter")
    label: str                                         # display ("OpenRouter")
    aliases: Tuple[str, ...] = ()                      # ("or",) for openrouter
    env_var_prefix: str = "ATLAS"                      # env var namespace

    # Endpoints
    base_url: str = ""
    chat_path: str = "/chat/completions"
    messages_path: str = "/messages"
    models_path: str = "/models"

    # Auth
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    key_prefix: str = ""

    # Behaviour
    capabilities: ProviderCapability = ProviderCapability.NONE
    pool_mode: PoolMode = PoolMode.PARTIAL_STICKY

    # Key file paths
    key_file: str = ""
    fallback_key_file: str = ""
    dead_keys_file: str = ""

    # Default model (used when client omits ``model``)
    default_model: str = ""

    # Per-provider extra upstream headers (e.g. OpenRouter rankings)
    extra_headers: Dict[str, str] = field(default_factory=dict)

    # Optional hook: classify an upstream response as a quota/rate-limit
    # error that should retire the key. ``None`` means "no provider-
    # specific check; trust status code only".
    is_quota_error: Optional[Callable[[int, Optional[bytes]], bool]] = None

    # Optional hook: classify as "key itself is dead" (vs rate-limited).
    is_key_dead: Optional[Callable[[int, Optional[bytes]], bool]] = None

    # ----- derived properties -----------------------------------------
    @property
    def chat_url(self) -> str:
        return f"{self.base_url}{self.chat_path}"

    @property
    def messages_url(self) -> str:
        return f"{self.base_url}{self.messages_path}"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}{self.models_path}"

    def has(self, cap: ProviderCapability) -> bool:
        return cap in self.capabilities

    def auth_header_value(self, key: str) -> str:
        """Build the value for the auth header. Bearer by default."""
        if not key:
            return ""
        if self.auth_scheme:
            return f"{self.auth_scheme} {key}"
        return key

    def check_quota(self, status: int, body: Optional[bytes]) -> bool:
        if self.is_quota_error is None:
            return False
        try:
            return bool(self.is_quota_error(status, body))
        except Exception as e:  # hook is best-effort
            log.warning("provider=%s is_quota_error raised: %s", self.name, e)
            return False

    def check_key_dead(self, status: int, body: Optional[bytes]) -> bool:
        if self.is_key_dead is None:
            return False
        try:
            return bool(self.is_key_dead(status, body))
        except Exception as e:
            log.warning("provider=%s is_key_dead raised: %s", self.name, e)
            return False


# ---------------------------------------------------------------------------
# Built-in providers
# ---------------------------------------------------------------------------

# HF quota markers (exposed for tests + the HF provider's hooks)
_HF_QUOTA_MARKERS: Tuple[str, ...] = (
    "rate limit reached",
    "quota exceeded",
    "credit balance is insufficient",
    "insufficient credits",
    "credits exhausted",
    "usage limit reached",
    "rate_limited",
)


def _is_hf_quota(status: int, body: Optional[bytes]) -> bool:
    """HF-specific quota/credit-exhaustion classifier.

    429 (standard) or 402 (Payment Required) plus a body scan for
    HF's quota markers. Used to permanently retire the key.
    """
    if status in (429, 402):
        return True
    if body:
        try:
            text = body.decode("utf-8", errors="ignore").lower()
            return any(m in text for m in _HF_QUOTA_MARKERS)
        except Exception:
            pass
    return False


def _is_hf_key_dead(status: int, body: Optional[bytes]) -> bool:
    """HF-specific dead-key classifier.

    401 with 'invalid'/'unauthorized', or 403 with 'invalid-api-key'/
    'revoked'. Generic 403 (model access denied) is NOT a dead key.
    """
    if not body:
        return False
    try:
        text = body.decode("utf-8", errors="ignore").lower()
    except Exception:
        return False
    if status == 401 and ("invalid" in text or "unauthorized" in text):
        return True
    if status == 403 and ("invalid-api-key" in text or "revoked" in text):
        return True
    return False


def _build_openrouter() -> Provider:
    """OpenRouter: full OpenAI-compat + a /messages endpoint.

    Note: the /messages endpoint is a recent OpenRouter addition; we
    advertise it but route the Anthropic-format request through the
    OpenAI chat/completions endpoint unless OPENROUTER_USE_MESSAGES=1
    is set in the future.
    """
    base = os.environ.get(
        "ATLAS_OPENROUTER_BASE_URL",
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    # Default to the same env-fallback chain config._load_or_default_model()
    # uses, so a fresh install with no runtime_provider.json never lands
    # on a dead OpenRouter slug (z-ai/glm-5.2:free 404s on OpenRouter).
    default_model = os.environ.get(
        "ATLAS_OPENROUTER_MODEL",
        os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m3:free"),
    )
    return Provider(
        name="openrouter",
        label="OpenRouter",
        aliases=("or",),
        base_url=base,
        key_prefix="sk-",
        # OpenRouter has both endpoints, but we route /v1/messages
        # through the chat endpoint with OpenAI->Anthropic translation
        # unless an explicit MESSAGES-capable proxy is configured.
        capabilities=ProviderCapability.OPENAI_COMPAT,
        pool_mode=PoolMode.PARTIAL_STICKY,
        default_model=default_model,
    )


def _build_huggingface() -> Provider:
    """HuggingFace router: OpenAI-compat with HF-specific error bodies.

    Uses full_sticky because alternating keys doesn't help when the
    whole upstream is rate-limited by IP/model. Has quota body
    markers -- the response body says "rate limit reached" rather
    than the standard 429-with-empty-body.
    """
    base = os.environ.get("ATLAS_HF_BASE_URL", "https://router.huggingface.co/v1")
    default_model = os.environ.get(
        "ATLAS_HF_MODEL", "deepseek-ai/DeepSeek-V4-Flash:deepinfra"
    )
    # The provider-specific key files come from config.py -- see the
    # post-init hook in config.py where these are patched in.
    return Provider(
        name="huggingface",
        label="HuggingFace",
        aliases=("hf",),
        base_url=base,
        key_prefix="hf_",
        capabilities=(
            ProviderCapability.OPENAI_COMPAT
            | ProviderCapability.HF_QUOTA_BODY_MARKERS
            | ProviderCapability.POOL_FULL_STICKY
        ),
        pool_mode=PoolMode.FULL_STICKY,
        default_model=default_model,
        is_quota_error=_is_hf_quota,
        is_key_dead=_is_hf_key_dead,
    )


def _build_nvidia() -> Provider:
    """NVIDIA NIM: OpenAI-compatible inference, no key required.

    NVIDIA's NIM endpoints use the `nvapi-` key prefix and the
    `/v1/chat/completions` endpoint. The key file is patched in
    by ``proxy.config`` after construction (like HF_CONFIG).
    """
    return Provider(
        name="nvidia",
        label="NVIDIA",
        aliases=("nv",),
        base_url=os.environ.get(
            "ATLAS_NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        ),
        key_prefix="nvapi-",
        capabilities=ProviderCapability.OPENAI_COMPAT,
        pool_mode=PoolMode.PARTIAL_STICKY,
        default_model="meta/llama-3.1-70b-instruct",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class _Registry:
    """Internal provider registry. Singleton.

    Stores providers in registration order so ``list_providers()`` is
    stable (useful for ``/v1/models`` and prettylog label maps).
    """
    _providers: Dict[str, Provider] = {}
    _order: List[str] = []
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(cls, provider: Provider, *, allow_override: bool = False) -> None:
        if provider.name in cls._providers and not allow_override:
            raise ValueError(
                f"Provider {provider.name!r} already registered. "
                f"Pass allow_override=True to replace it."
            )
        if provider.name in cls._providers:
            # Override -- rebuild alias map after the swap
            old = cls._providers[provider.name]
            for a in old.aliases:
                cls._aliases.pop(a, None)
        cls._providers[provider.name] = provider
        if provider.name not in cls._order:
            cls._order.append(provider.name)
        for a in provider.aliases:
            if a in cls._aliases and cls._aliases[a] != provider.name:
                raise ValueError(
                    f"Alias {a!r} already maps to {cls._aliases[a]!r}; "
                    f"cannot remap to {provider.name!r}"
                )
            cls._aliases[a] = provider.name

    @classmethod
    def get(cls, name: str) -> Optional[Provider]:
        return cls._providers.get(name)

    @classmethod
    def by_alias(cls, token: str) -> Optional[Provider]:
        canon = cls._aliases.get(token.lower())
        if canon is not None:
            return cls._providers.get(canon)
        return cls._providers.get(token)

    @classmethod
    def names(cls) -> List[str]:
        return list(cls._order)

    @classmethod
    def labels(cls) -> Dict[str, str]:
        return {n: cls._providers[n].label for n in cls._order if n in cls._providers}

    @classmethod
    def all(cls) -> Dict[str, Provider]:
        return dict(cls._providers)


def register_provider(
    provider: Provider, *, allow_override: bool = False
) -> Provider:
    """Add a provider to the registry.

    >>> from proxy.providers import Provider, ProviderCapability, register_provider
    >>> nvidia = Provider(
    ...     name="nvidia",
    ...     label="NVIDIA NIM",
    ...     base_url="https://integrate.api.nvidia.com/v1",
    ...     key_prefix="nvapi-",
    ...     default_model="meta/llama-3.1-70b-instruct",
    ...     capabilities=ProviderCapability.OPENAI_COMPAT,
    ... )
    >>> register_provider(nvidia)
    """
    _Registry.register(provider, allow_override=allow_override)
    return provider


def get_provider(name: Optional[str] = None) -> Optional[Provider]:
    """Resolve a provider by name. ``None`` returns the active provider.

    Returns ``None`` if the name is unknown. Use ``get_provider_or_404``
    if you want a hard error.
    """
    if name is None:
        name = _ACTIVE_PROVIDER_NAME
    if not name:
        return None
    return _Registry.get(name) or _Registry.by_alias(name)


def get_active_provider() -> Provider:
    """Return the active provider, or fall back to openrouter."""
    p = get_provider(None)
    if p is not None:
        return p
    # Last-resort: openrouter must be registered by config.py at startup
    fallback = _Registry.get("openrouter")
    if fallback is None:
        raise RuntimeError(
            "No providers registered. proxy.config must be imported first."
        )
    return fallback


def list_providers() -> List[str]:
    """Canonical names in registration order."""
    return _Registry.names()


def provider_labels() -> Dict[str, str]:
    """Canonical name -> display label map. Used by prettylog & /v1/models."""
    return _Registry.labels()


def resolve_provider_name(token: str) -> Optional[str]:
    """Map an alias (``"or"``, ``"hf"``) to the canonical name. Case-insensitive.

    Returns ``None`` if the alias is unknown. Useful when reading
    user-supplied config (CLI flags, JSON, env vars).
    """
    if not token:
        return None
    p = _Registry.by_alias(token)
    return p.name if p else None


# The active provider is set by ``proxy.config`` after the env vars
# and runtime config file have been read. Until then, ``None``.
_ACTIVE_PROVIDER_NAME: Optional[str] = None


def set_active_provider_name(name: Optional[str]) -> None:
    """Called by ``proxy.config`` once the active provider is known.

    Public for testing: tests can swap the active provider to verify
    behaviour without rebooting the process.
    """
    global _ACTIVE_PROVIDER_NAME
    _ACTIVE_PROVIDER_NAME = name


def get_active_provider_name() -> Optional[str]:
    return _ACTIVE_PROVIDER_NAME


# ---------------------------------------------------------------------------
# Backwards-compat: keep the old name available
# ---------------------------------------------------------------------------

# Old code used ``ProviderConfig`` -- the new ``Provider`` is a strict
# superset, so the alias is safe.
ProviderConfig = Provider


__all__ = [
    "Provider",
    "ProviderCapability",
    "ProviderConfig",  # back-compat alias
    "PoolMode",
    "register_provider",
    "get_provider",
    "get_active_provider",
    "get_active_provider_name",
    "set_active_provider_name",
    "list_providers",
    "provider_labels",
    "resolve_provider_name",
    "_is_hf_quota",
    "_is_hf_key_dead",
]
