"""Key pool — lock-free atomic round-robin + health management."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    COOLDOWN_BASE_SECONDS,
    COOLDOWN_MAX_SECONDS,
    HF_KEY_PREFIX,
    MAX_CONSECUTIVE_ERRORS,
    MAX_CONCURRENT_PER_KEY,
    RETRY_STATUSES,
    STICKY_MAX_USES,
    SUSPEND_SECONDS,
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
    sticky_uses: int = 0
    last_status: int = 0
    last_latency_ms: float = 0.0
    in_flight: int = 0


class KeyPool:
    """
    Atomic round-robin key selector with per-key cooldown, exponential backoff,
    temporary suspension after repeated failures, and automatic recovery.

    Selection advances a monotonic index (GIL-safe for simple ints on CPython).
    Health mutations are protected by an asyncio.Lock.

    Modes:
      - "partial_sticky" (default): OpenRouter behaviour. Stays on a key while
        healthy and under the per-key concurrency cap, then rotates.
      - "full_sticky": Hugging Face behaviour. Always returns the same key
        until it is explicitly retired via retire_key(). Never rotates among
        healthy keys. Retired keys are permanently excluded.
    """

    def __init__(self, keys: List[str], mode: str = "partial_sticky") -> None:
        self.mode: str = mode
        self._sticky_idx: Optional[int] = None
        self._retired: set[int] = set()
        if not keys:
            log.info("KeyPool initialized with 0 keys (will auto-load)")
            self._keys: List[KeyInfo] = []
            self._n = 0
            self._idx = 0
            self._lock = asyncio.Lock()
            return
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

    async def reload_keys(self, new_keys: List[str]) -> Tuple[int, int, int]:
        """Reload keys from new list, preserving state of existing keys.

        Returns (added, removed, kept) counts.

        Key identity is tracked by *key string*, not index.  This means:
          - Retired keys stay retired even if their index shifts.
          - The sticky key is preserved only if the same key string is
            still present and healthy.
          - Reordering / insertion / removal of other keys cannot corrupt
            retired or sticky state.

        Empty ``new_keys`` is a no-op: we never discard a healthy existing
        pool because of a transient empty read.
        """
        if not new_keys:
            log.warning("reload_keys called with empty list — keeping existing pool")
            return (0, 0, self._n)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for k in new_keys:
            if k not in seen:
                seen.add(k)
                unique.append(k)

        async with self._lock:
            # Snapshot current state keyed by string
            old_by_key: Dict[str, KeyInfo] = {k.key: k for k in self._keys}
            old_retired_keys: set[str] = {
                self._keys[i].key for i in self._retired if 0 <= i < self._n
            }
            old_sticky_key: Optional[str] = None
            if self._sticky_idx is not None and 0 <= self._sticky_idx < self._n:
                old_sticky_key = self._keys[self._sticky_idx].key

            # Build new list, preserving KeyInfo for keys that remain
            new_key_info: List[KeyInfo] = []
            for i, key in enumerate(unique):
                if key in old_by_key:
                    info = old_by_key[key]
                    info.index = i
                    new_key_info.append(info)
                else:
                    new_key_info.append(KeyInfo(key=key, index=i))

            # Rebuild retired set by key string
            self._retired: set[int] = {
                i for i, info in enumerate(new_key_info) if info.key in old_retired_keys
            }

            # Restore sticky only if the same key is still present and not retired
            self._sticky_idx: Optional[int] = None
            if old_sticky_key is not None:
                for i, info in enumerate(new_key_info):
                    if info.key == old_sticky_key and i not in self._retired:
                        self._sticky_idx = i
                        break

            old_n = self._n
            self._keys = new_key_info
            self._n = len(self._keys)
            # Clamp ring position
            if self._idx >= self._n:
                self._idx = 0

            added = sum(1 for k in unique if k not in old_by_key)
            removed = old_n - sum(1 for k in old_by_key if k in seen)
            kept = self._n - added
            return (added, removed, kept)

    @property
    def total(self) -> int:
        return self._n

    def next_key(self) -> Tuple[str, int, bool]:
        """
        Return the next key for a request — partial-sticky + round-robin.

        Returns (key, index, is_healthy):
        - key: the API key string
        - index: the key's position in the pool (for in-flight accounting)
        - is_healthy: True if the key was selected as healthy (normal path).
          False if this is a Phase-3 fallback (all keys cooling/suspended) —
          the caller should consider fast-failing rather than burning retries
          against a key that will almost certainly fail.

        Selection strategy:
        - If a "sticky" key is set and still healthy and under the per-key
          concurrency cap, return it (stays pinned for up to
          MAX_CONCURRENT_PER_KEY in-flight requests).
        - Otherwise, advance the sticky pointer via round-robin among healthy
          keys that have capacity (< cap in-flight).
        - If no healthy key has capacity, still return the next healthy key
          (its in-flight may exceed the cap, but we must not block the request).
        - If no healthy keys exist at all, return the next key anyway with
          is_healthy=False so the caller can fast-fail.

        Selection never blocks waiting for a slot — the caller's in_flight
        accounting + cooldown handles backpressure at a higher level.
        """
        if self._n == 0:
            raise ValueError("No keys available in pool")

        # --- Full-sticky mode (Hugging Face) ---
        # Always return the current sticky key unless it has been retired.
        # If the sticky key is retired or doesn't exist, pick the first
        # non-retired key and make it sticky.
        if self.mode == "full_sticky":
            now = time.monotonic()
            # Auto-recover cooling/suspended keys whose timers expired
            for info in self._keys:
                if (
                    info.state in (KeyState.COOLING, KeyState.SUSPENDED)
                    and info.cooldown_until <= now
                ):
                    info.state = KeyState.HEALTHY
                    info.consecutive_errors = 0

            if self._sticky_idx is not None:
                sticky_info = self._keys[self._sticky_idx]
                if (
                    self._sticky_idx not in self._retired
                    and sticky_info.state == KeyState.HEALTHY
                ):
                    return sticky_info.key, self._sticky_idx, True

            # Sticky key is retired/dead/nonexistent — pick next non-retired key
            for i in range(self._n):
                if i not in self._retired and self._keys[i].state == KeyState.HEALTHY:
                    self._sticky_idx = i
                    return self._keys[i].key, i, True

            # All keys retired or unhealthy — return next non-retired (may be unhealthy)
            for i in range(self._n):
                if i not in self._retired:
                    self._sticky_idx = i
                    return self._keys[i].key, i, (self._keys[i].state == KeyState.HEALTHY)

            raise ValueError("All keys retired in full_sticky pool")

        # --- Partial-sticky / round-robin mode (OpenRouter — existing logic) ---
        now = time.monotonic()
        start = self._idx

        # --- Auto-recover cooling/suspended keys whose timers expired ---
        for info in self._keys:
            if (
                info.state in (KeyState.COOLING, KeyState.SUSPENDED)
                and info.cooldown_until <= now
            ):
                info.state = KeyState.HEALTHY
                info.consecutive_errors = 0

        # --- Phase 1: stickiness check ---
        # If we have a sticky key, try to keep using it while it's healthy
        # and below the concurrency cap.
        if self._sticky_idx is not None:
            sticky_info = self._keys[self._sticky_idx]
            if (
                sticky_info.state == KeyState.HEALTHY
                and sticky_info.in_flight < MAX_CONCURRENT_PER_KEY
                and sticky_info.sticky_uses < STICKY_MAX_USES
            ):
                return sticky_info.key, self._sticky_idx, True
            # Sticky key is unhealthy, at capacity, or worn out its welcome —
            # fall through to rotation (a little push to the next key).

        # --- Phase 2: round-robin rotation ---
        # Walk the ring starting from _idx, preferring healthy keys with capacity.
        # If none have capacity, prefer healthy keys regardless of capacity.
        # If none healthy, fall back to the plain round-robin next key.
        best_i: int | None = None
        best_inflight = 10**9
        capacity_found = False

        for _ in range(self._n):
            i = self._idx % self._n
            self._idx = i + 1
            info = self._keys[i]

            if info.state == KeyState.HEALTHY:
                if info.in_flight < MAX_CONCURRENT_PER_KEY:
                    # Has capacity — prefer this over at-capacity keys
                    capacity_found = True
                    if info.in_flight < best_inflight:
                        best_inflight = info.in_flight
                        best_i = i
                        if best_inflight == 0:
                            break  # perfect candidate
                elif not capacity_found and info.in_flight < best_inflight:
                    # At capacity but no capacity-bearing key found yet — fallback
                    best_inflight = info.in_flight
                    best_i = i

        if best_i is not None:
            self._sticky_idx = best_i
            self._keys[best_i].sticky_uses = 0  # fresh lease on the new sticky key
            return self._keys[best_i].key, self._keys[best_i].index, True

        # --- Phase 3: fallback — all keys cooling/suspended ---
        i = start % self._n
        self._idx = i + 1
        if self._sticky_idx != i:
            self._keys[i].sticky_uses = 0
        self._sticky_idx = i
        return self._keys[i].key, self._keys[i].index, False

    async def acquire(self, index: int) -> None:
        """Increment in-flight count when a request starts using this key."""
        async with self._lock:
            self._keys[index].in_flight += 1

    async def release(self, index: int) -> None:
        """Decrement in-flight count when a request finishes (success or fail)."""
        async with self._lock:
            info = self._keys[index]
            info.in_flight = max(0, info.in_flight - 1)

    async def mark_success(self, index: int, latency_ms: float = 0.0) -> None:
        async with self._lock:
            info = self._keys[index]
            info.state = KeyState.HEALTHY
            info.consecutive_errors = 0
            info.cooldown_until = 0.0
            info.total_requests += 1
            if index == self._sticky_idx:
                info.sticky_uses += 1
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

    async def retire_key(self, index: int) -> None:
        """Permanently retire a key (full-sticky mode only).

        Marks the key as SUSPENDED with no cooldown recovery and adds it
        to the _retired set so next_key() will never select it again.
        The caller is responsible for persisting the key to dead_hf_keys.txt.
        """
        async with self._lock:
            if index < 0 or index >= self._n:
                return
            self._retired.add(index)
            info = self._keys[index]
            info.state = KeyState.SUSPENDED
            info.cooldown_until = float("inf")  # never recovers
            log.warning(
                "Key idx=%d permanently retired (key=%s..., retired_count=%d)",
                index,
                info.key[:20],
                len(self._retired),
            )

    def is_key_retired(self, index: int) -> bool:
        """Check if a key index has been permanently retired."""
        return index in self._retired

    def get_retired_key_strings(self) -> List[str]:
        """Return the actual key strings for retired indices."""
        return [self._keys[i].key for i in sorted(self._retired) if 0 <= i < self._n]

    def stats(self) -> Dict[str, Any]:
        now = time.monotonic()
        healthy = cooling = suspended = 0
        in_flight = 0
        for info in self._keys:
            in_flight += info.in_flight
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
            "in_flight": in_flight,
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
                    "in_flight": info.in_flight,
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


_KEY_PREFIXES = ("sk-", "sk-ant-", "sk-proj-", "sk-openai-", HF_KEY_PREFIX)


def load_keys(path: str) -> List[str]:
    """Load API keys from a text file.

    Recognizes lines starting with any known key prefix (sk-, sk-ant-, sk-proj-,
    sk-openai-, hf_). Ignores blanks, comments, and non-key lines.
    Tolerates "KEY=sk-…" or plain key formats.
    """
    if not os.path.isfile(path):
        return []
    keys: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # tolerate "KEY=sk-…" or plain key
            if "=" in line and not line.startswith(_KEY_PREFIXES):
                line = line.split("=", 1)[-1].strip()
            if line.startswith(_KEY_PREFIXES):
                keys.append(line)
    return keys
