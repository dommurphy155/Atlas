"""
Provider registry for managing multiple providers.

This module provides a centralized registry for managing provider instances,
including:
- Provider registration
- Provider retrieval
- Key rotation support
- Provider fallback
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.providers.base import Provider, ProviderConfig, ProviderError


@dataclass
class KeyStoreEntry:
    """An API key with its metadata."""
    key: str
    index: int
    cooldown_until: float = 0.0

    def is_available(self) -> bool:
        """Check if key is available (not in cooldown)."""
        return time.monotonic() >= self.cooldown_until


class KeyStore:
    """
    Simple key store with cooldown support.

    Manages API keys with automatic cooldown when keys fail.
    """

    def __init__(self, keys: list[str], cooldown_seconds: float = 60.0):
        self._keys = [KeyStoreEntry(key=k, index=i) for i, k in enumerate(keys)]
        self._cooldown_seconds = cooldown_seconds
        self._active_index: int = -1
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return len(self._keys) > 0

    def get_available_key(self) -> Optional[tuple[str, int]]:
        """Get an available key without setting it as active."""
        now = time.monotonic()

        # Try active key first
        if self._active_index >= 0 and self._active_index < len(self._keys):
            entry = self._keys[self._active_index]
            if entry.is_available():
                return entry.key, entry.index

        # Search for next available key
        for i, entry in enumerate(self._keys):
            if entry.is_available():
                return entry.key, entry.index

        # No available keys
        return None

    def acquire(self) -> Optional[tuple[str, int]]:
        """Acquire a key for use. Returns (key, index) or None."""
        now = time.monotonic()

        # If active key is available, keep using it (sticky behavior)
        if self._active_index >= 0 and self._active_index < len(self._keys):
            entry = self._keys[self._active_index]
            if entry.is_available():
                return entry.key, entry._index

        # Find next available key
        for i, entry in enumerate(self._keys):
            if entry.is_available():
                self._active_index = i
                return entry.key, entry.index

        # All keys in cooldown
        return None

    def cooldown(self, index: int) -> None:
        """Mark a key as in cooldown."""
        if 0 <= index < len(self._keys):
            self._keys[index].cooldown_until = time.monotonic() + self._cooldown_seconds


@dataclass
class ProviderEntry:
    """A provider with its configuration."""
    name: str
    provider: Provider
    priority: int = 0  # Higher = tried first
    enabled: bool = True


class ProviderRegistry:
    """
    Registry for managing multiple providers with fallback support.

    Supports:
    - Multiple providers with priority ordering
    - Key rotation within providers
    - Automatic fallback on failure
    - Runtime provider enable/disable
    """

    def __init__(self):
        self._providers: dict[str, ProviderEntry] = {}
        self._lock = asyncio.Lock()
        self._key_stores: dict[str, KeyStore] = {}

    def register(
        self,
        name: str,
        provider: Provider,
        priority: int = 0,
        enabled: bool = True,
    ) -> None:
        """Register a provider."""
        self._providers[name] = ProviderEntry(
            name=name,
            provider=provider,
            priority=priority,
            enabled=enabled,
        )

    def unregister(self, name: str) -> None:
        """Unregister a provider."""
        if name in self._providers:
            del self._providers[name]

    def get(self, name: str) -> Optional[Provider]:
        """Get a provider by name."""
        entry = self._providers.get(name)
        return entry.provider if entry else None

    def get_enabled_providers(self) -> list[Provider]:
        """Get all enabled providers sorted by priority."""
        enabled = [e for e in self._providers.values() if e.enabled]
        return [e.provider for e in sorted(enabled, key=lambda e: -e.priority)]

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a provider."""
        if name in self._providers:
            self._providers[name].enabled = enabled

    def get_key_store(self, provider_name: str) -> Optional[KeyStore]:
        """Get the key store for a provider."""
        return self._key_stores.get(provider_name)

    def register_key_store(self, provider_name: str, keys: list[str], cooldown_seconds: float = 60.0) -> None:
        """Register a key store for a provider."""
        self._key_stores[provider_name] = KeyStore(keys, cooldown_seconds)

    async def acquire_key(self, provider_name: str) -> Optional[tuple[str, int]]:
        """Acquire an API key for a provider."""
        store = self._key_stores.get(provider_name)
        if store:
            return store.acquire()
        return None

    async def cooldown_key(self, provider_name: str, index: int) -> None:
        """Mark a key as in cooldown."""
        store = self._key_stores.get(provider_name)
        if store:
            store.cooldown(index)

    async def close_all(self) -> None:
        """Close all providers."""
        for entry in self._providers.values():
            try:
                await entry.provider.close()
            except Exception:
                pass


# Global registry instance
_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None


# Convenience functions
def register_provider(name: str, provider: Provider, priority: int = 0) -> None:
    """Register a provider in the global registry."""
    get_registry().register(name, provider, priority)


def get_provider(name: str) -> Optional[Provider]:
    """Get a provider from the global registry."""
    return get_registry().get(name)


def get_all_providers() -> list[Provider]:
    """Get all enabled providers from the global registry."""
    return get_registry().get_enabled_providers()
