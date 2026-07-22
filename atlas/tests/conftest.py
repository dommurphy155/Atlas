"""Shared pytest fixtures + helpers for the Atlas regression suite.

Atlas's modules resolve paths from their own ``__file__`` and from the
``ATLAS_STATS_DIR`` / ``ATLAS_STATS_FILE`` env vars, so tests isolate state by
pointing those at a tmp_path and by monkeypatching the module-level path
constants. No test touches the live ``data/`` directory or the live NVIDIA API.

Async support: we avoid a hard dependency on pytest-asyncio so the suite runs
on a bare venv with only pytest installed. Each async test is written as a sync
function that calls ``run(coro)`` — a thin wrapper around asyncio.run. (If
pytest-asyncio IS installed, the @pytest.mark.asyncio tests in the suite also
work; both styles are supported.)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Make the repo importable as `proxy.*` (mirrors how the service launches:
# `python -m proxy.atlas_proxy` from the project root).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proxy import stats as stats_mod  # noqa: E402


def run(coro):
    """Run a coroutine to completion synchronously. Used by async tests so
    the suite doesn't require pytest-asyncio."""
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(agen):
    """Drain an async generator into a list. Lets sync tests consume an async
    generator via collect() without writing an async comprehension (which is a
    SyntaxError in a sync function frame)."""
    out = []
    async for chunk in agen:
        out.append(chunk)
    return out


def collect(agen):
    """Sync wrapper: drain an async generator to a list from a sync test."""
    return run(_collect(agen))


@pytest.fixture(autouse=True)
def isolated_stats(tmp_path, monkeypatch):
    """Redirect stats persistence to a tmp dir so tests never clobber the
    real data/proxy_stats.json. Re-points the module-level STATS_DIR/STATS_FILE
    and the ATLAS_STATS_DIR env var so token_tracker (which reads the env var
    at import time) sees the same file."""
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    stats_file = stats_dir / "proxy_stats.json"
    monkeypatch.setenv("ATLAS_STATS_DIR", str(stats_dir))
    monkeypatch.setattr(stats_mod, "STATS_DIR", stats_dir)
    monkeypatch.setattr(stats_mod, "STATS_FILE", stats_file)
    yield stats_file


@pytest.fixture
def override_file(tmp_path, monkeypatch):
    """A writable override file in tmp_path, wired into system_prompt via its
    module-level OVERRIDE_PATH constant."""
    from proxy import system_prompt as sp

    path = tmp_path / "system_prompt_override.txt"
    monkeypatch.setattr(sp, "OVERRIDE_PATH", path)
    # Reset the mtime cache so each test re-reads from its own file.
    monkeypatch.setattr(sp, "_override_cache", None)
    monkeypatch.setattr(sp, "_override_mtime", None)
    return path


@pytest.fixture
def keys_file(tmp_path):
    """An empty keys.txt in tmp_path. Tests write keys into it."""
    path = tmp_path / "keys.txt"
    path.write_text("")
    return path
