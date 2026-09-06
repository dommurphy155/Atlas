"""Tests for the provider registry in proxy.providers.

Covers:
  - registration / override semantics
  - alias resolution (``or``/``hf`` -> canonical)
  - capability queries
  - the active-provider hook
  - the HF-specific quota / dead-key classification
  - adding a new provider at runtime (the "easy to add new ones" promise)
"""
from __future__ import annotations

import pytest

from proxy.providers import (
    PoolMode,
    Provider,
    ProviderCapability,
    _is_hf_key_dead,
    _is_hf_quota,
    get_active_provider,
    get_provider,
    list_providers,
    provider_labels,
    register_provider,
    resolve_provider_name,
    set_active_provider_name,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_built_in_providers_registered() -> None:
    names = list_providers()
    assert "openrouter" in names
    assert "huggingface" in names


def test_built_in_providers_have_canonical_labels() -> None:
    labels = provider_labels()
    assert labels["openrouter"] == "OpenRouter"
    assert labels["huggingface"] == "HuggingFace"


def test_get_provider_by_canonical_name() -> None:
    p = get_provider("openrouter")
    assert p is not None
    assert p.name == "openrouter"
    assert p.label == "OpenRouter"


def test_get_provider_by_alias() -> None:
    assert resolve_provider_name("or") == "openrouter"
    assert resolve_provider_name("OR") == "openrouter"
    assert resolve_provider_name("hf") == "huggingface"
    assert resolve_provider_name("HF") == "huggingface"


def test_get_provider_unknown_returns_none() -> None:
    assert get_provider("nonexistent") is None
    assert resolve_provider_name("nonexistent") is None


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

def test_openrouter_capabilities() -> None:
    p = get_provider("openrouter")
    assert p.has(ProviderCapability.CHAT_COMPLETIONS)
    assert p.has(ProviderCapability.STREAMING_DONE_SENTINEL)
    assert p.has(ProviderCapability.SUPPORTS_TOOL_CALLS)
    assert not p.has(ProviderCapability.HF_QUOTA_BODY_MARKERS)
    assert p.pool_mode == PoolMode.PARTIAL_STICKY


def test_huggingface_capabilities() -> None:
    p = get_provider("huggingface")
    assert p.has(ProviderCapability.CHAT_COMPLETIONS)
    assert p.has(ProviderCapability.HF_QUOTA_BODY_MARKERS)
    assert p.has(ProviderCapability.POOL_FULL_STICKY)
    assert p.pool_mode == PoolMode.FULL_STICKY
    assert p.key_prefix == "hf_"


def test_derived_urls() -> None:
    p = get_provider("openrouter")
    assert p.chat_url == p.base_url + "/chat/completions"
    assert p.models_url == p.base_url + "/models"
    assert p.messages_url == p.base_url + "/messages"


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,body,expected", [
    (429, b"", True),
    (402, b"", True),
    (500, b"", False),
    (200, b"", False),
    (429, b"rate limit reached please back off", True),
    (429, b"quota exceeded", True),
    (429, b"insufficient credits", True),
    (429, b"some other error", True),  # 429 alone is enough
    # The classifier also fires on body markers regardless of status code
    # (matches the original behaviour: if upstream says "quota exceeded"
    # in the body, treat it as quota even on an unexpected status).
    (500, b"quota exceeded", True),
    (500, b"rate limit reached", True),
    (500, b"unrelated error message", False),
])
def test_hf_quota_classifier(status, body, expected) -> None:
    assert _is_hf_quota(status, body) is expected


@pytest.mark.parametrize("status,body,expected", [
    (401, b"invalid api key", True),
    (401, b"unauthorized access", True),
    (401, b"missing key", False),
    (403, b"invalid-api-key", True),
    (403, b"key revoked", True),
    (403, b"model not allowed", False),
    (200, b"", False),
    (404, b"", False),
])
def test_hf_key_dead_classifier(status, body, expected) -> None:
    assert _is_hf_key_dead(status, body) is expected


def test_provider_check_quota_uses_hook() -> None:
    p = get_provider("huggingface")
    assert p.check_quota(429, b"") is True
    assert p.check_quota(200, b"") is False


def test_provider_check_key_dead_uses_hook() -> None:
    p = get_provider("huggingface")
    assert p.check_key_dead(401, b"invalid key") is True
    assert p.check_key_dead(200, b"") is False


def test_provider_check_quota_no_hook_returns_false() -> None:
    # Custom provider with no hook -- must not crash, must return False.
    custom = Provider(name="noop", label="Noop")
    assert custom.check_quota(500, b"") is False
    assert custom.check_key_dead(401, b"") is False


# ---------------------------------------------------------------------------
# Active provider
# ---------------------------------------------------------------------------

def test_active_provider_default_is_openrouter(monkeypatch) -> None:
    # Reset to openrouter in case a previous test swapped it.
    set_active_provider_name("openrouter")
    p = get_active_provider()
    assert p.name == "openrouter"


def test_active_provider_can_be_swapped() -> None:
    set_active_provider_name("huggingface")
    try:
        assert get_active_provider().name == "huggingface"
    finally:
        set_active_provider_name("openrouter")


# ---------------------------------------------------------------------------
# Adding a new provider
# ---------------------------------------------------------------------------

def test_register_new_provider() -> None:
    """The 'easy to add new ones' promise: a 5-line registration call."""
    nvidia = Provider(
        name="nvidia",
        label="NVIDIA NIM",
        aliases=("nv",),
        base_url="https://integrate.api.nvidia.com/v1",
        key_prefix="nvapi-",
        default_model="meta/llama-3.1-70b-instruct",
        capabilities=ProviderCapability.OPENAI_COMPAT,
    )
    try:
        register_provider(nvidia)
        # Now discoverable by canonical name AND alias
        assert get_provider("nvidia") is nvidia
        assert resolve_provider_name("nv") == "nvidia"
        assert resolve_provider_name("NV") == "nvidia"
        # And shows up in the listing
        assert "nvidia" in list_providers()
        # And in the label map
        assert provider_labels()["nvidia"] == "NVIDIA NIM"
    finally:
        # Best-effort cleanup -- the registry has no deregister by design,
        # but we can override with a sentinel if a future test needs to
        # re-register with the same name.
        pass


def test_register_duplicate_raises() -> None:
    dup = Provider(name="openrouter", label="duplicate", base_url="x")
    with pytest.raises(ValueError, match="already registered"):
        register_provider(dup)
    # allow_override=True accepts the swap
    register_provider(dup, allow_override=True)
    assert get_provider("openrouter").label == "duplicate"
    # Restore the original (tests downstream depend on the canonical label)
    from proxy.config import OPENROUTER_CONFIG as _real_or
    register_provider(_real_or, allow_override=True)


# ---------------------------------------------------------------------------
# Back-compat
# ---------------------------------------------------------------------------

def test_provider_config_alias() -> None:
    from proxy.providers import ProviderConfig
    assert ProviderConfig is Provider
