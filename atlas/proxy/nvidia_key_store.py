from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any


# Sticky NVIDIA key pool with per-key cooldown + per-key concurrency limit.
#
# Behavior:
#   - Each key has a max concurrent request limit (default 8). acquire() skips
#     keys at capacity, picks the next eligible key. This spreads concurrent
#     load across the pool instead of hammering one key.
#   - When a key hits rate-limit / quota / auth / transport error, caller calls
#     cooldown_key(), which blacklists the key for COOLDOWN_SECONDS (default 60s).
#   - The next acquire() scans forward from the cooled key's position, picks the
#     next eligible (non-cooled, under-capacity) key as the new sticky active key.
#   - A key whose cooldown has expired does NOT preempt the current active key.
#     It simply becomes eligible again the next time the rotation naturally
#     reaches it.
#   - When a request completes (success or failure), caller calls release_key()
#     to decrement the in-flight counter.
#
# Keys are loaded from disk (one per line) and reload on mtime change, so the
# operator can edit data/keys.txt live and the pool picks it up. Keys are never
# permanently removed.
class NvidiaKeyStore:
    def __init__(
        self,
        keys_file: str,
        reload_seconds: int = 5,
        cooldown_seconds: float = 60.0,
        max_concurrent_per_key: int = 8,
    ) -> None:
        self.keys_file = Path(keys_file)
        self.reload_seconds = max(1, reload_seconds)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.max_concurrent_per_key = max(1, max_concurrent_per_key)
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
        # key fingerprint -> in-flight request count
        self._in_flight: dict[str, int] = {}

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
        """A key is eligible iff it exists, not on cooldown, and under concurrency limit."""
        if idx < 0 or idx >= len(self._keys):
            return False
        key = self._keys[idx]
        if self._cooling_until(key) > now:
            return False
        if self._in_flight.get(key, 0) >= self.max_concurrent_per_key:
            return False
        return True

    async def acquire(self) -> tuple[str, int] | None:
        """Sticky acquire with per-key concurrency limit.

        - If the active key is eligible (exists, not on cooldown, under capacity),
          return it again. This is the hot path: repeated requests reuse the same
          key until that key fails or hits its concurrency limit.
        - If the active key is ineligible (cooled, at capacity, or unset), scan
          forward through the pool from the active position and stick to the first
          eligible key we find. That key becomes the new sticky active key.
        - If every key is ineligible, return None so the caller 503s instead of
          hammering a cooled/overloaded key.

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
                key = self._keys[idx]
                self._in_flight[key] = self._in_flight.get(key, 0) + 1
                return key, idx

            # Active key is ineligible — scan forward for the next eligible
            # key and make it the new sticky active key. Start the scan at the
            # active index (or 0 if none) so we resume the forward rotation in
            # place rather than jumping back to the top of the list.
            start = (self._active_index + 1) % n if self._active_index >= 0 else 0
            for offset in range(n):
                idx = (start + offset) % n
                if self._is_eligible(idx, now):
                    self._active_index = idx
                    key = self._keys[idx]
                    self._in_flight[key] = self._in_flight.get(key, 0) + 1
                    return key, idx

            # Every key is ineligible (cooled or at capacity). Never reuse a
            # blacklisted/overloaded key — that defeats the cooldown and creates
            # a 429 retry loop. Return None so the caller 503s.
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

    async def release_key(self, key: str) -> None:
        """Decrement the in-flight counter for a key.

        Call this when a request completes (success or failure) so the key's
        concurrency slot is freed for the next acquire().
        """
        async with self._lock:
            if key in self._in_flight:
                self._in_flight[key] = max(0, self._in_flight[key] - 1)
                if self._in_flight[key] == 0:
                    del self._in_flight[key]

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        cooling = sum(1 for until in self._cooldowns.values() if until > now)
        active_valid = 0 <= self._active_index < len(self._keys)
        in_flight_total = sum(self._in_flight.values())
        # Show top keys by in-flight count (fingerprinted, not raw keys)
        top_keys = sorted(
            [(fingerprint(k, self._keys.index(k) if k in self._keys else None), v)
             for k, v in self._in_flight.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        # Fingerprinted in_flight_by_key for debugging
        in_flight_by_key = {
            fingerprint(k, self._keys.index(k) if k in self._keys else None): v
            for k, v in self._in_flight.items()
        }
        return {
            "total_keys": len(self._keys),
            "available": len(self._keys) > 0,
            "cooling_down": cooling,
            "active_key_index": self._active_index,
            "active_key_eligible": active_valid and self._cooling_until(self._keys[self._active_index]) <= now,
            "in_flight_total": in_flight_total,
            "in_flight_by_key": in_flight_by_key,
            "top_in_flight_keys": top_keys,
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
