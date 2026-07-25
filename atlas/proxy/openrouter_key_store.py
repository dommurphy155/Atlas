from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any


# Sticky OpenRouter key pool with per-key cooldown.
#
# Same behavior as NvidiaKeyStore:
#   - Exactly one key is "active" at any time.
#   - Every request uses the active key, repeatedly, until that key returns a
#     rate-limit / quota / auth / transport error.
#   - On such a failure the caller calls cooldown_key(), which blacklists the
#     active key for COOLDOWN_SECONDS (default 60s).
#   - The next acquire() then scans forward from the cooled key's position,
#     picks the next eligible (non-cooled) key, and that becomes the new sticky
#     active key.
#   - A key whose cooldown has expired does NOT preempt the current active key.
#     It simply becomes eligible again the next time the rotation naturally
#     reaches it — i.e. only when the active key eventually fails and the scan
#     walks past it.
#
# Keys are loaded from disk (one per line) and reload on mtime change, so the
# operator can edit data/openroute_keys.txt live and the pool picks it up. Keys
# are never permanently removed.
class OpenRouterKeyStore:
    def __init__(self, keys_file: str, reload_seconds: int = 5, cooldown_seconds: float = 60.0) -> None:
        self.keys_file = Path(keys_file)
        self.reload_seconds = max(1, reload_seconds)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._keys: list[str] = []
        self._active_index: int = -1
        self._lock = asyncio.Lock()
        self._mtime: float | None = None
        self._cooldowns: dict[str, float] = {}

    @property
    def available(self) -> bool:
        return len(self._keys) > 0

    async def load(self, force: bool = False) -> None:
        async with self._lock:
            self.keys_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.keys_file.exists():
                self.keys_file.touch(mode=0o600)

            try:
                mtime = self.keys_file.stat().st_mtime
            except FileNotFoundError:
                self._keys = []
                return

            if not force and self._mtime == mtime:
                return

            raw = self.keys_file.read_text().splitlines()
            seen: set[str] = set()
            keys: list[str] = []
            for line in raw:
                token = line.strip()
                if token and token not in seen:
                    seen.add(token)
                    keys.append(token)

            self._keys = keys
            self._mtime = mtime
            if self._active_index >= len(self._keys):
                self._active_index = -1

    async def reload_if_changed(self) -> None:
        await self.load(force=False)

    async def watch(self) -> None:
        while True:
            await asyncio.sleep(self.reload_seconds)
            await self.reload_if_changed()

    async def acquire(self) -> tuple[str, int] | None:
        """Return an eligible key with its index, or None if no keys available."""
        await self.reload_if_changed()
        async with self._lock:
            if not self._keys:
                return None

            now = time.monotonic()
            if self._active_index < 0:
                # First acquisition: start at 0
                self._active_index = 0

            # Scan forward from current position for an eligible key
            attempts = 0
            while attempts < len(self._keys):
                idx = (self._active_index + attempts) % len(self._keys)
                key = self._keys[idx]
                cooldown_until = self._cooldowns.get(key, 0.0)

                if cooldown_until <= now:
                    self._active_index = idx
                    return (key, idx)

                attempts += 1

            # All keys are on cooldown; still return the one with earliest expiry
            # so the caller can fail fast and trigger a retry sooner.
            earliest_key = min(
                self._keys,
                key=lambda k: self._cooldowns.get(k, 0.0)
            )
            idx = self._keys.index(earliest_key)
            self._active_index = idx
            return (earliest_key, idx)

    async def cooldown_key(self, key: str) -> None:
        """Blacklist a key for cooldown_seconds."""
        async with self._lock:
            self._cooldowns[key] = time.monotonic() + self.cooldown_seconds

    def stats(self) -> dict[str, Any]:
        """Return a stats dict for /stats."""
        now = time.monotonic()
        keys_on_cooldown = sum(
            1 for k, until in self._cooldowns.items()
            if until > now and k in self._keys
        )
        return {
            "total_keys": len(self._keys),
            "keys_on_cooldown": keys_on_cooldown,
            "active_key": self._keys[self._active_index] if 0 <= self._active_index < len(self._keys) else None,
        }


def fingerprint(key: str, index: int | None = None) -> str:
    """Short, leak-safe key identity for logs: ``#idx(…last4)``."""
    tail = key[-6:] if len(key) >= 6 else key
    if index is not None:
        return f"#{index}(…{tail})"
    return f"…{tail}"
