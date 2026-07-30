"""Key pool — lock-free atomic round-robin + health management."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Tuple

from .config import (
    COOLDOWN_BASE_SECONDS,
    COOLDOWN_MAX_SECONDS,
    MAX_CONSECUTIVE_ERRORS,
    RETRY_STATUSES,
    SUSPEND_SECONDS,
    get_keys_file,
    get_fallback_keys_file,
    get_logger,
)

log = get_logger(__name__)


class KeyState(IntEnum):
    HEALTHY = 0
    COOLING = 1
    SUSPENDED = 2


@dataclass(slots=True)
class KeyInfo:
    key: str
    index: int
    state: KeyState = KeyState.HEALTHY
    cooldown_until: float = 0.0
    consecutive_errors: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_status: int = 0
    last_latency_ms: float = 0.0


class KeyPool:
    """
    Atomic round-robin key selector with per-key cooldown, exponential backoff,
    temporary suspension after repeated failures, and automatic recovery.

    Selection advances a monotonic index (GIL-safe for simple ints on CPython).
    Health mutations are protected by an asyncio.Lock.
    """

    def __init__(self, keys: List[str]) -> None:
        if not keys:
            raise ValueError("No API keys loaded")
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        self._keys: List[KeyInfo] = [
            KeyInfo(key=k, index=i) for i, k in enumerate(unique)
        ]
        self._n = len(self._keys)
        self._idx = 0
        self._lock = asyncio.Lock()

    @property
    def total(self) -> int:
        return self._n

    def next_key(self) -> Tuple[str, int]:
        """
        Return the next healthy key, advancing the global index.
        Auto-recovers keys whose cooldown/suspension has expired.
        If every key is unavailable, still returns the next one so the
        caller can attempt the request and mark the error.
        """
        now = time.monotonic()
        start = self._idx
        for _ in range(self._n):
            i = self._idx % self._n
            self._idx = i + 1
            info = self._keys[i]
            if info.state == KeyState.HEALTHY:
                return info.key, info.index
            if (
                info.state in (KeyState.COOLING, KeyState.SUSPENDED)
                and info.cooldown_until <= now
            ):
                info.state = KeyState.HEALTHY
                info.consecutive_errors = 0
                return info.key, info.index
        # All cooling/suspended — return next anyway
        i = start % self._n
        self._idx = i + 1
        return self._keys[i].key, self._keys[i].index

    async def mark_success(self, index: int, latency_ms: float = 0.0) -> None:
        async with self._lock:
            info = self._keys[index]
            info.state = KeyState.HEALTHY
            info.consecutive_errors = 0
            info.cooldown_until = 0.0
            info.total_requests += 1
            info.last_status = 200
            info.last_latency_ms = latency_ms

    async def mark_error(self, index: int, status: int) -> None:
        async with self._lock:
            info = self._keys[index]
            info.consecutive_errors += 1
            info.total_errors += 1
            info.total_requests += 1
            info.last_status = status

            if info.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                info.state = KeyState.SUSPENDED
                info.cooldown_until = time.monotonic() + SUSPEND_SECONDS
                log.warning(
                    "Key idx=%d suspended for %.0fs after %d consecutive errors (status=%s)",
                    index,
                    SUSPEND_SECONDS,
                    info.consecutive_errors,
                    status,
                )
            elif status in RETRY_STATUSES:
                backoff = min(
                    COOLDOWN_BASE_SECONDS * (2 ** (info.consecutive_errors - 1)),
                    COOLDOWN_MAX_SECONDS,
                )
                info.state = KeyState.COOLING
                info.cooldown_until = time.monotonic() + backoff
                log.warning(
                    "Key idx=%d cooldown %.0fs (status=%s, consecutive=%d)",
                    index,
                    backoff,
                    status,
                    info.consecutive_errors,
                )

    def stats(self) -> Dict[str, Any]:
        now = time.monotonic()
        healthy = cooling = suspended = 0
        for info in self._keys:
            if info.state == KeyState.HEALTHY or (
                info.state in (KeyState.COOLING, KeyState.SUSPENDED)
                and info.cooldown_until <= now
            ):
                healthy += 1
            elif info.state == KeyState.COOLING:
                cooling += 1
            else:
                suspended += 1
        return {
            "total": self._n,
            "healthy": healthy,
            "cooling": cooling,
            "suspended": suspended,
        }

    def detailed_stats(self) -> List[Dict[str, Any]]:
        now = time.monotonic()
        out: List[Dict[str, Any]] = []
        for info in self._keys:
            state = info.state.name
            if info.state != KeyState.HEALTHY and info.cooldown_until <= now:
                state = "HEALTHY (recovered)"
            out.append(
                {
                    "index": info.index,
                    "state": state,
                    "consecutive_errors": info.consecutive_errors,
                    "total_requests": info.total_requests,
                    "total_errors": info.total_errors,
                    "last_status": info.last_status,
                    "last_latency_ms": round(info.last_latency_ms, 1),
                    "cooldown_remaining_s": max(
                        0.0, round(info.cooldown_until - now, 1)
                    ),
                }
            )
        return out


def load_keys(path: str) -> List[str]:
    """Load keys from a text file. Ignores blanks, comments, and non-sk-/nvapi- lines."""
    if not os.path.isfile(path):
        return []
    keys: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # tolerate "KEY=sk-…" or "KEY=nvapi-…" or plain key
            if "=" in line and not (line.startswith("sk-") or line.startswith("nvapi-")):
                line = line.split("=", 1)[-1].strip()
            if line.startswith("sk-") or line.startswith("nvapi-"):
                keys.append(line)
    return keys
