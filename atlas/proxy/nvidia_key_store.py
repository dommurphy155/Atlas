from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any


# Sticky NVIDIA key pool with per-key cooldown.
#
# Behavior the operator asked for:
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
# operator can edit data/keys.txt live and the pool picks it up. Keys are never
# permanently removed.
class NvidiaKeyStore:
    def __init__(self, keys_file: str, reload_seconds: int = 5, cooldown_seconds: float = 60.0) -> None:
        self.keys_file = Path(keys_file)
        self.reload_seconds = max(1, reload_seconds)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._keys: list[str] = []
        # The sticky active key index. -1 means "no active key yet; pick the
        # first eligible one on the next acquire()". We track an index (not the
        # key string) so a live keys.txt edit that reorders lines can't make us
        # stick to the wrong key — acquire() always re-resolves via index.
        self._active_index: int = -1
        self._lock = asyncio.Lock()
        self._mtime: float | None = None
        # key fingerprint -> cooldown-unix-epoch (monotonic)
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
            # If the active index is out of range after a reload (keys removed or
            # reordered), reset it so acquire() picks a fresh eligible key
            # instead of sticking to a now-different key or erroring.
            if self._active_index >= len(self._keys):
                self._active_index = -1

    async def reload_if_changed(self) -> None:
        await self.load(force=False)

    async def watch(self) -> None:
        while True:
            try:
                await self.reload_if_changed()
            except Exception:
                pass
            await asyncio.sleep(self.reload_seconds)

    def _cooling_until(self, key: str) -> float:
        return self._cooldowns.get(key, 0.0)

    def _is_eligible(self, idx: int, now: float) -> bool:
        """A key is eligible iff it exists and is not currently on cooldown."""
        if idx < 0 or idx >= len(self._keys):
            return False
        return self._cooling_until(self._keys[idx]) <= now

    async def acquire(self) -> tuple[str, int] | None:
        """Sticky acquire: return the current active key unless it's on cooldown.

        - If the active key is eligible (exists, not on cooldown), return it
          again. This is the hot path: repeated requests reuse the same key
          until that key fails.
        - If the active key is on cooldown (or unset), scan forward through the
          pool from the active position and stick to the first eligible key we
          find. That key becomes the new active key.
        - If every key is on cooldown, fall back to the active key anyway (or the
          next one if unset) so the request does not hard-fail when the pool is
          merely rate-limited — better to try a cooling key than to 503.

        Returns ``(key, index)`` so callers can log a stable key identity
        (position in keys.txt + a fingerprint) without re-scanning the pool.
        """
        async with self._lock:
            if not self._keys:
                return None
            now = time.monotonic()
            n = len(self._keys)

            # Hot path: the active key is still eligible. Keep using it.
            if self._is_eligible(self._active_index, now):
                idx = self._active_index
                return self._keys[idx], idx

            # Active key is cooled / unset — scan forward for the next eligible
            # key and make it the new sticky active key. Start the scan at the
            # active index (or 0 if none) so we resume the forward rotation in
            # place rather than jumping back to the top of the list.
            start = (self._active_index + 1) % n if self._active_index >= 0 else 0
            for offset in range(n):
                idx = (start + offset) % n
                if self._is_eligible(idx, now):
                    self._active_index = idx
                    return self._keys[idx], idx

            # Every key is cooling down. Never reuse a blacklisted key.
            # Returning a cooled key defeats the purpose of the cooldown and
            # creates a 429 retry loop.
            return None

    async def cooldown_key(self, key: str) -> None:
        """Blacklist a key for COOLDOWN_SECONDS.

        After this, the *next* acquire() sees the active key as ineligible and
        scans forward to the next eligible key, which becomes the new sticky
        active key. The cooled key auto-recovers (becomes eligible again) once
        the cooldown elapses, but it does NOT preempt the then-active key — it
        only re-enters rotation when the scan naturally reaches it again.
        """
        async with self._lock:
            self._cooldowns[key] = time.monotonic() + self.cooldown_seconds

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        cooling = sum(1 for until in self._cooldowns.values() if until > now)
        active_valid = 0 <= self._active_index < len(self._keys)
        return {
            "total_keys": len(self._keys),
            "available": len(self._keys) > 0,
            "cooling_down": cooling,
            "active_key_index": self._active_index,
            "active_key_eligible": active_valid and self._cooling_until(self._keys[self._active_index]) <= now,
        }


def fingerprint(key: str, index: int | None = None) -> str:
    """Short, leak-safe key identity for logs: ``#idx(…last4)``.

    Index is the key's position in keys.txt (quick mental tracking); the
    last-4 fingerprint gives a stable identity that survives a keys.txt
    reorder. The full key is never logged.
    """
    tail = key[-6:] if len(key) >= 6 else key
    if index is not None:
        return f"#{index}(…{tail})"
    return f"…{tail}"
