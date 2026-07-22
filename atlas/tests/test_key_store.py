"""NvidiaKeyStore — sticky rotation, cooldown, hot-reload.

The store is the most safety-critical pure logic in the proxy: it decides
which live credential serves each request. These tests pin its documented
contract (README §Keys, §Architecture resilience loop):

  - exactly one key is "active" and reused until it fails
  - a failed key is cooled for COOLDOWN_SECONDS then auto-recovers
  - the keys file is hot-reloaded on mtime change
  - acquire() never returns a cooled key; if all are cooling it returns None
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from proxy.nvidia_key_store import NvidiaKeyStore, fingerprint


def _write_keys(path: Path, keys: list[str]) -> None:
    path.write_text("\n".join(keys) + ("\n" if keys else ""))
    # Bump mtime so the store's watch() notices the change even on filesystems
    # with coarse (1s) mtime granularity.
    st = os.stat(path)
    os.utime(path, (st.st_atime, st.st_mtime + 2))


def _make(path: Path, cooldown: float = 60.0, reload_seconds: int = 1) -> NvidiaKeyStore:
    return NvidiaKeyStore(str(path), reload_seconds=reload_seconds, cooldown_seconds=cooldown)


KEYS = [f"nvapi-key{i:02d}aaaa" for i in range(6)]


@pytest.mark.asyncio
async def test_load_and_acquire_sticky(keys_file):
    """The same key is handed out repeatedly until it fails — the hot path."""
    _write_keys(keys_file, KEYS)
    store = _make(keys_file)
    await store.load(force=True)

    first = await store.acquire()
    assert first is not None
    key, idx = first
    assert key == KEYS[0]
    assert idx == 0

    # Repeated acquire returns the SAME sticky key (not round-robin).
    for _ in range(5):
        again = await store.acquire()
        assert again == (key, idx)


@pytest.mark.asyncio
async def test_cooldown_rotates_to_next_key(keys_file):
    """Cooling the active key makes acquire() scan forward to the next eligible."""
    _write_keys(keys_file, KEYS)
    store = _make(keys_file)
    await store.load(force=True)

    active, _ = await store.acquire()
    assert active == KEYS[0]

    await store.cooldown_key(active)
    nxt = await store.acquire()
    assert nxt is not None
    assert nxt[0] == KEYS[1], "after cooldown, the next key becomes sticky"


@pytest.mark.asyncio
async def test_cooled_key_is_not_reused(keys_file):
    """A cooled key must not be handed back out while on cooldown."""
    _write_keys(keys_file, KEYS)
    store = _make(keys_file)
    await store.load(force=True)

    k0, _ = await store.acquire()
    await store.cooldown_key(k0)
    # Every subsequent acquire must skip k0.
    for _ in range(10):
        got = await store.acquire()
        assert got is not None
        assert got[0] != k0


@pytest.mark.asyncio
async def test_all_cooled_returns_none(keys_file):
    """When every key is on cooldown, acquire() returns None (per the code's
    explicit 'never reuse a blacklisted key' branch) — NOT a cooled key."""
    _write_keys(keys_file, KEYS)
    store = _make(keys_file)
    await store.load(force=True)

    for k in KEYS:
        await store.cooldown_key(k)

    assert await store.acquire() is None


@pytest.mark.asyncio
async def test_cooldown_auto_expires(keys_file):
    """A cooled key becomes eligible again once cooldown_seconds elapses."""
    _write_keys(keys_file, KEYS)
    store = _make(keys_file, cooldown=0.05)
    await store.load(force=True)

    k0, _ = await store.acquire()
    await store.cooldown_key(k0)
    nxt = await store.acquire()
    assert nxt[0] != k0

    await asyncio.sleep(0.06)
    # After expiry the cooled key (k0) is eligible again. To prove the scan
    # actually reaches a *recovered* key — not just any eligible one — cool
    # every other key so the only path forward wraps back to k0. (The scan
    # resumes forward from the active position; without cooling the rest it
    # would hit k2 and never wrap, falsely passing on a key that was never
    # cooled.) k0 is the recovered one we want to see handed back.
    for k in KEYS:
        if k != k0:
            await store.cooldown_key(k)
    recovered = await store.acquire()
    assert recovered is not None
    assert recovered[0] == k0, "after cooldown expiry, the recovered key must be eligible again"


@pytest.mark.asyncio
async def test_hot_reload_picks_up_new_keys(keys_file):
    """Editing keys.txt live adds keys to the pool without a restart."""
    _write_keys(keys_file, KEYS[:3])
    store = _make(keys_file)
    await store.load(force=True)
    assert (await store.acquire())[0] == KEYS[0]

    _write_keys(keys_file, KEYS)  # add 3 more
    await store.reload_if_changed()

    # The pool now has 6 keys; cool the first 3 and the scan should reach a
    # newly-added key, proving reload took effect.
    for k in KEYS[:3]:
        await store.cooldown_key(k)
    got = await store.acquire()
    assert got is not None
    assert got[0] in KEYS[3:]


@pytest.mark.asyncio
async def test_hot_reload_dedupes(keys_file):
    """Duplicate lines in keys.txt collapse to one pool entry."""
    _write_keys(keys_file, [KEYS[0], KEYS[0], KEYS[1], KEYS[1]])
    store = _make(keys_file)
    await store.load(force=True)
    assert store.stats()["total_keys"] == 2


@pytest.mark.asyncio
async def test_empty_keys_file_returns_none(keys_file):
    """An empty keys file yields no keys; acquire() returns None cleanly."""
    _write_keys(keys_file, [])
    store = _make(keys_file)
    await store.load(force=True)
    assert store.available is False
    assert await store.acquire() is None


@pytest.mark.asyncio
async def test_stats_reports_cooling_count(keys_file):
    _write_keys(keys_file, KEYS)
    store = _make(keys_file)
    await store.load(force=True)

    await store.cooldown_key(KEYS[0])
    await store.cooldown_key(KEYS[1])
    s = store.stats()
    assert s["total_keys"] == 6
    assert s["cooling_down"] == 2
    assert s["available"] is True


def test_fingerprint_never_leaks_full_key():
    """Log fingerprints must never contain the full credential."""
    full = "nvapi-0123456789abcdef"
    fp = fingerprint(full, index=2)
    assert full not in fp
    assert fp.startswith("#2(")
    assert fp.endswith(")")
    # No-idx variant also safe.
    assert full not in fingerprint(full)
