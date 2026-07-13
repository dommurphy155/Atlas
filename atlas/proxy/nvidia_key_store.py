from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any


# Round-robin NVIDIA key pool with per-key cooldown.
#
# Keys are loaded from disk (one per line) and rotate indefinitely. A key that
# fails upstream (429/401/403/5xx) is put on a short cooldown so it is skipped
# for COOLDOWN_SECONDS rather than retried every rotation. Keys are never
# permanently removed — they reload from disk on mtime change, so the operator
# can edit data/keys.txt live and the pool picks it up.
class NvidiaKeyStore:
    def __init__(self, keys_file: str, reload_seconds: int = 5, cooldown_seconds: float = 30.0) -> None:
        self.keys_file = Path(keys_file)
        self.reload_seconds = max(1, reload_seconds)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._keys: list[str] = []
        self._next_index = 0
        self._lock = asyncio.Lock()
        self._mtime: float | None = None
        # key fingerprint -> cooldown-unix-epoch
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
            if self._next_index >= len(self._keys):
                self._next_index = 0

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

    async def acquire(self) -> tuple[str, int] | None:
        """Round-robin acquire, skipping keys that are on cooldown.

        Returns ``(key, index)`` so callers can log a stable key identity
        (position in keys.txt + a fingerprint) without re-scanning the pool.
        """
        async with self._lock:
            if not self._keys:
                return None
            now = time.monotonic()
            n = len(self._keys)
            for offset in range(n):
                idx = (self._next_index + offset) % n
                key = self._keys[idx]
                if self._cooling_until(key) > now:
                    continue
                self._next_index = (idx + 1) % n
                return key, idx
            # Every key is on cooldown — fall back to the next one anyway so the
            # request does not hard-fail when the pool is merely rate-limited.
            key = self._keys[self._next_index]
            idx = self._next_index
            self._next_index = (self._next_index + 1) % n
            return key, idx

    async def cooldown_key(self, key: str) -> None:
        async with self._lock:
            self._cooldowns[key] = time.monotonic() + self.cooldown_seconds

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        cooling = sum(1 for until in self._cooldowns.values() if until > now)
        return {
            "total_keys": len(self._keys),
            "available": len(self._keys) > 0,
            "cooling_down": cooling,
        }


def fingerprint(key: str, index: int | None = None) -> str:
    """Short, leak-safe key identity for logs: ``#idx(…last4)``.

    Index is the key's position in keys.txt (quick mental tracking); the
    last-4 fingerprint gives a stable identity that survives a keys.txt
    reorder. The full key is never logged.
    """
    tail = key[-4:] if len(key) >= 4 else key
    if index is not None:
        return f"#{index}(…{tail})"
    return f"…{tail}"
