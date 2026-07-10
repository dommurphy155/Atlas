"""Key pool: load from disk, round-robin acquire, cooldown on 429, disable on auth/credits.

The file format is one key per line. Lines that are blank or don't start with
`nvapi-` or `nvda-` are ignored. Keys are deduped. The file is reloaded on a
mtime poll so edits show up without a restart.

Three states:
  - usable:          not disabled, cooldown expired
  - cooling_down:    429'd, will be usable again after cooldown_until
  - disabled:        401/403/402, removed from the file atomically (permanent)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

# Accept NVIDIA's real nvapi- keys and the nvda- shape from the spec.
_VALID_PREFIXES = ("nvapi-", "nvda-")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"{token[:4]}...{digest[:10]}"


def _is_valid_line(token: str) -> bool:
    return bool(token) and token.startswith(_VALID_PREFIXES)


@dataclass
class KeyState:
    token: str
    fingerprint: str
    total_requests: int = 0
    failed_requests: int = 0
    last_used: str | None = None
    cooldown_until: float = 0.0
    disabled_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.disabled_reason is None and time.time() >= self.cooldown_until

    @property
    def cooling_down(self) -> bool:
        return self.disabled_reason is None and time.time() < self.cooldown_until

    def public(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "last_used": self.last_used,
            "cooldown_until": (
                datetime.fromtimestamp(self.cooldown_until, timezone.utc).isoformat()
                if self.cooldown_until
                else None
            ),
            "disabled_reason": self.disabled_reason,
        }


class KeyStore:
    def __init__(self, keys_file: str, reload_seconds: int) -> None:
        self.keys_file = Path(keys_file)
        self.reload_seconds = max(1, reload_seconds)
        self._keys: list[KeyState] = []
        self._known_stats: dict[str, KeyState] = {}
        self._mtime: float | None = None
        self._last_reload: str | None = None
        self._next_index = 0
        self._lock = asyncio.Lock()
        self._disabled_runtime = 0

    @property
    def last_reload(self) -> str | None:
        return self._last_reload

    @property
    def keys_mtime(self) -> str | None:
        try:
            return datetime.fromtimestamp(
                self.keys_file.stat().st_mtime, timezone.utc
            ).isoformat()
        except FileNotFoundError:
            return None

    async def load(self, force: bool = False) -> None:
        # Refresh the in-memory key list from disk. Dedupe keeps the file sane
        # even if duplicate tokens were appended by accident. Stats for known
        # keys survive a reload so cooldowns aren't lost on a file touch.
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

            raw_tokens = [
                line.strip()
                for line in self.keys_file.read_text().splitlines()
                if line.strip()
            ]
            seen: set[str] = set()
            tokens: list[str] = []
            for token in raw_tokens:
                if not _is_valid_line(token) or token in seen:
                    continue
                seen.add(token)
                tokens.append(token)

            next_keys: list[KeyState] = []
            for token in tokens:
                fp = fingerprint(token)
                state = self._known_stats.get(fp)
                if state is None:
                    state = KeyState(token=token, fingerprint=fp)
                else:
                    state.token = token
                next_keys.append(state)
                self._known_stats[fp] = state

            self._keys = next_keys
            self._mtime = mtime
            self._last_reload = utc_now_iso()
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

    async def acquire(self) -> KeyState | None:
        # Round-robin through usable keys instead of hammering the first one.
        async with self._lock:
            if not self._keys:
                return None

            total = len(self._keys)
            for offset in range(total):
                idx = (self._next_index + offset) % total
                state = self._keys[idx]
                if state.available:
                    self._next_index = idx
                    state.total_requests += 1
                    state.last_used = utc_now_iso()
                    # Advance past the chosen key so the next acquire starts fresh.
                    self._next_index = (self._next_index + 1) % total
                    return state
            return None

    async def fail_key(self, state: KeyState) -> None:
        # 5xx / transient: bump the fail counter, keep the key in rotation.
        async with self._lock:
            state.failed_requests += 1

    async def cooldown_key(self, state: KeyState, seconds: int) -> None:
        # 429: put it on ice, retry once the upstream rate-limit window moves.
        async with self._lock:
            state.failed_requests += 1
            state.cooldown_until = time.time() + max(1, seconds)

    async def disable_key(self, state: KeyState, reason: str) -> None:
        # 401/403/402: dead credential or out of credits. Remove from the file
        # so the next reload doesn't bring it back. Permanent.
        async with self._lock:
            state.failed_requests += 1
            state.disabled_reason = reason
            self._disabled_runtime += 1
            active_tokens = [k.token for k in self._keys if k is not state]
            self._atomic_write_tokens(active_tokens)
            self._keys = [k for k in self._keys if k is not state]
            self._mtime = self.keys_file.stat().st_mtime
            self._last_reload = utc_now_iso()
            if self._next_index >= len(self._keys):
                self._next_index = 0

    def _atomic_write_tokens(self, tokens: list[str]) -> None:
        # Write through a temp file + os.replace so a crash mid-write can't
        # corrupt keys.txt.
        payload = "".join(f"{token}\n" for token in tokens)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.keys_file.name}.",
            suffix=".tmp",
            dir=str(self.keys_file.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w") as tmp:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self.keys_file)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def stats(self) -> dict[str, Any]:
        usable = sum(1 for k in self._keys if k.available)
        cooling = sum(1 for k in self._keys if k.cooling_down)
        disabled = sum(1 for k in self._keys if k.disabled_reason)
        return {
            "total_keys": len(self._keys),
            "usable_keys": usable,
            "cooling_down": cooling,
            "disabled": disabled,
            "disabled_this_runtime": self._disabled_runtime,
        }

    def key_stats(self) -> list[dict[str, Any]]:
        return [k.public() for k in self._keys]


# Module-level singleton, instantiated lazily so tests can build their own.
_key_store: KeyStore | None = None


def get_key_store() -> KeyStore:
    global _key_store
    if _key_store is None:
        _key_store = KeyStore(config.KEYS_FILE, config.RELOAD_SECONDS)
    return _key_store
