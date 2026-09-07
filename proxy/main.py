"""FastAPI app, lifespan, and uvicorn entrypoint.

Run:
  python -m proxy.main
"""

from __future__ import annotations

import sys
import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Set by main() after _pick_port() resolves the actual bind port (which may
# differ from the configured LISTEN_PORT if 8777 is occupied). Read by the
# lifespan startup log so it reports the same port uvicorn actually bound
# to — prevents the "INFO listening on 8777 / uvicorn on 57027" mismatch.
_BOUND_PORT: int = 0

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import routes
from .config import (
    CORS_ORIGINS,
    FALLBACK_KEY_FILE,
    KEY_FILE,
    KEEPALIVE_EXPIRY,
    LISTEN_HOST,
    LISTEN_PORT,
    LOG_LEVEL,
    SYSTEM_PROMPT_OVERRIDE_FILE,
    get_default_model,
    get_force_default_model,
    reload_system_prompt_override,
    migrate_hf_active_keys,
    log,
)
from .providers import (
    ProviderCapability,
    get_active_provider,
)
from .keypool import KeyPool, load_keys
from .proxy import ProxyCore
from . import prettylog as _prettylog

# Logging-only: swap the pretty formatter onto the existing root handler(s),
# then mirror every line into proxy/logs/atlas-proxy.log (plain text, rotated).
_prettylog.attach()
_prettylog.mirror_file()

# Optional high-performance event loop
try:
    import uvloop

    uvloop.install()
except ImportError:
    pass


def _load_provider_keys() -> list[str]:
    """Load keys from the active provider's key file.

    For HuggingFace, loads ``hf_keys.txt`` directly. Dead keys are
    never in the active file because ``retire_and_remove_hf_key()``
    removes them on retirement, and ``migrate_hf_active_keys()`` cleans
    up any orphans at startup. For OpenRouter, returns keys from
    ``KEY_FILE`` unchanged (preserving existing behaviour).
    """
    provider = get_active_provider()
    return load_keys(provider.key_file)


# Public alias retained for the test-suite / external callers.
_load_active_keys = _load_provider_keys


def _reload_keys_for_provider() -> list[str]:
    """Reload keys, respecting dead-key exclusion for HF and the
    OpenRouter fallback file when the primary is empty."""
    provider = get_active_provider()
    if provider.has(ProviderCapability.HF_QUOTA_BODY_MARKERS):
        return _load_provider_keys()
    # OpenRouter-style provider: use fallback if primary is empty
    keys = load_keys(provider.key_file)
    if not keys:
        keys = load_keys(FALLBACK_KEY_FILE)
        if keys:
            log.warning(
                "Primary keys file missing -- using fallback %s (%d keys)",
                FALLBACK_KEY_FILE,
                len(keys),
            )
    return keys


@asynccontextmanager
async def lifespan(app: FastAPI):
    provider = get_active_provider()

    # --- Startup migration: clean dead keys from HF active file ---
    if provider.has(ProviderCapability.HF_QUOTA_BODY_MARKERS):
        removed, kept = migrate_hf_active_keys()
        log.info("HF key migration: removed %d dead keys, kept %d active", removed, kept)

    # --- Shuffle key file in-place before pool construction (startup only) ---
    # Randomises key order so fresh boots don't hammer the same keys in
    # sequence after repeated local testing/restarts.
    # Pure-Python (replaces `shuf`) so it works on macOS too.
    _key_file = KEY_FILE
    try:
        import random
        with open(_key_file) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        random.shuffle(lines)
        tmp = Path(str(_key_file) + ".tmp")
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, _key_file)
    except FileNotFoundError:
        pass  # no key file yet -- pool starts empty and hot-reloads
    except Exception as e:
        log.warning("key-file shuffle failed (non-fatal): %s", e)

    keys = _reload_keys_for_provider()

    log.info(
        "Loaded %d %s keys (mode=%s)",
        len(keys),
        provider.name,
        provider.pool_mode.value,
    )

    if not keys:
        log.warning(
            "No keys found yet. Expected file: %s. "
            "Proxy will start and auto-load keys as they become available.",
            provider.key_file or KEY_FILE,
        )
        keys = []  # Start with empty pool, will be populated

    pool = KeyPool(keys, mode=provider.pool_mode.value) if keys else KeyPool(
        [], mode=provider.pool_mode.value
    )

    # For OpenRouter-style providers, preserve the fallback behaviour
    if not provider.has(ProviderCapability.HF_QUOTA_BODY_MARKERS) and not keys:
        fallback_keys = load_keys(FALLBACK_KEY_FILE)
        if fallback_keys:
            log.warning(
                "Primary keys file missing -- using fallback %s (%d keys)",
                FALLBACK_KEY_FILE,
                len(fallback_keys),
            )
            pool = KeyPool(fallback_keys)

    core = ProxyCore(pool, provider=provider.name)

    # Start background key reloader
    reload_task = asyncio.create_task(_reload_keys_periodically(pool))

    await core.start()
    routes.proxy = core
    log.info(
        "Proxy listening on http://%s:%d  (keys=%d, healthy=%d, "
        "provider=%s, default_model=%s, force=%s)",
        LISTEN_HOST,
        _BOUND_PORT or LISTEN_PORT,
        pool.stats()["total"],
        pool.stats()["healthy"],
        provider.name,
        get_default_model(),
        get_force_default_model(),
    )
    yield
    reload_task.cancel()
    await core.stop()
    routes.proxy = None
    log.info("Shutdown complete")


async def _reload_keys_periodically(pool: KeyPool) -> None:
    """Periodically reload keys from file and update pool.

    For OpenRouter, uses KEY_FILE (existing behaviour unchanged).
    For HuggingFace, uses HF_KEY_FILE while respecting dead_hf_keys.txt.
    Also hot-reloads system prompt override (shared).

    Change detection is content-based, not mtime-based: we compare the
    parsed key list against the pool's current keys.  This means rapid
    successive writes, preserved mtimes, or partial writes that happen
    to share a timestamp cannot leave the pool stale.
    """
    last_override_mtime = 0
    last_key_fingerprint: Optional[tuple] = None  # (file_path, tuple_of_keys)

    while True:
        try:
            await asyncio.sleep(5)  # Check every 5 seconds

            override_file = Path(SYSTEM_PROMPT_OVERRIDE_FILE)
            if override_file.exists():
                omtime = override_file.stat().st_mtime
                if omtime > last_override_mtime:
                    last_override_mtime = omtime
                    reload_system_prompt_override()

            # Provider-specific key file
            key_file_path = get_active_provider().key_file

            key_file = Path(key_file_path)
            if not key_file.exists():
                continue

            new_keys = _load_provider_keys()
            fingerprint = (key_file_path, tuple(new_keys))

            # Skip if nothing changed (content-based, not mtime)
            if fingerprint == last_key_fingerprint:
                continue
            last_key_fingerprint = fingerprint

            current_key_strs = [k.key for k in pool._keys]
            if new_keys and new_keys != current_key_strs:
                old_count = len(pool._keys)
                added, removed, kept = await pool.reload_keys(new_keys)
                log.info(
                    "Reloaded keys from %s (%d → %d keys, +%d/-%d/=%d kept)",
                    key_file_path,
                    old_count,
                    len(pool._keys),
                    added,
                    removed,
                    kept,
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("Error reloading keys: %s", e)


app = FastAPI(
    title="OpenRouter Translation Proxy",
    version="1.1.0",
    lifespan=lifespan,
)

_cors_allow_credentials = bool(CORS_ORIGINS) and "*" not in CORS_ORIGINS
# Browsers reject allow_credentials=True with wildcard origins (CORS spec).
# When CORS_ORIGINS is empty (no browser clients) or contains "*", disable
# credentials so the proxy does not serve a malformed Access-Control-Allow-Origin
# response.

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-request-id"],
)

app.include_router(routes.router)


def _force_free_port(port: int, *, max_attempts: int = 10) -> bool:
    """Ensure ``port`` is free for binding by identifying and killing any
    process holding it.

    Atlas MUST own its configured port (8777 by default). On restart the
    previous atlas-proxy process can linger (e.g. systemd kill timeout,
    leftover from a crashed run), and a stale occupant must NOT cause us
    to silently bind a random port — that leaves the old (stale) process
    serving requests while health checks ping the new random port.

    Strategy:
      1. ss/lsof lookup the PID holding the port.
      2. SIGTERM, wait up to 2s for it to die.
      3. SIGKILL if still alive.
      4. Retry the bind. After ``max_attempts`` failures, raise.
    """
    import os
    import signal
    import socket
    import subprocess
    import time

    def _bind_ok() -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    for attempt in range(max_attempts):
        if _bind_ok():
            return True

        # Find PID(s) on the port. `ss -tlnp` requires root for the PID
        # column; `fuser -n tcp PORT` works without root and prints PIDs.
        pids: set[int] = set()
        try:
            r = subprocess.run(
                ["fuser", "-n", "tcp", str(port)],
                capture_output=True, text=True, timeout=3,
            )
            # fuser prints PIDs space-separated on stdout; some distros
            # put them on stderr. Accept either.
            for stream in (r.stdout, r.stderr):
                for tok in stream.split():
                    if tok.isdigit():
                        pids.add(int(tok))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Fallback: parse ss output (works on Ubuntu where fuser may be absent).
        if not pids:
            try:
                r = subprocess.run(
                    ["ss", "-tlnp", f"sport = :{port}"],
                    capture_output=True, text=True, timeout=3,
                )
                for line in r.stdout.splitlines()[1:]:
                    for tok in line.split():
                        if tok.startswith("pid="):
                            try:
                                pids.add(int(tok[4:].rstrip(",)")))
                            except ValueError:
                                pass
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        if not pids:
            # No occupant we can identify — wait briefly and retry in case
            # something is in TIME_WAIT.
            time.sleep(0.25)
            continue

        for pid in pids:
            # Never kill ourselves or our parent process tree.
            if pid == os.getpid() or pid == os.getppid():
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError:
                # Not our process to kill — refuse rather than fail silently.
                raise PermissionError(
                    f"port {port} held by pid {pid} which we cannot kill"
                )

        # Wait up to 2s for graceful exit, then SIGKILL.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            alive = False
            for pid in pids:
                try:
                    os.kill(pid, 0)
                    alive = True
                except ProcessLookupError:
                    pass
            if not alive:
                break
            time.sleep(0.1)
        else:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(0.2)

    return _bind_ok()


def main() -> None:
    import uvicorn

    # Atlas MUST own its configured port. If something else (a stale
    # atlas-proxy process, a previous test run that crashed, etc.) is
    # holding LISTEN_PORT, identify and kill it before we bind. We never
    # silently fall back to a random port — that would leave the stale
    # process serving requests while the installer/health checks probe
    # the new random port, masking the real failure.
    if not _force_free_port(LISTEN_PORT):
        raise SystemExit(
            f"could not free port {LISTEN_PORT} after repeated attempts; "
            f"check what is holding it (e.g. `ss -tlnp sport = :{LISTEN_PORT}`) "
            f"and kill it manually"
        )
    chosen_port = LISTEN_PORT
    try:
        from pathlib import Path as _P
        _bound = _P(__file__).resolve().parent.parent / "data" / "bound_port"
        _bound.parent.mkdir(parents=True, exist_ok=True)
        _bound.write_text(str(chosen_port))
    except OSError:
        pass

    # Expose the actually-bound port to the lifespan startup log so it
    # agrees with uvicorn's own banner (which prints the bound port).
    global _BOUND_PORT
    _BOUND_PORT = chosen_port

    # Pass the app object, not "proxy.main:app" — the string form makes uvicorn
    # re-import this module while __main__ already ran it, double-executing all
    # module-level setup (incl. logging handlers).
    uvicorn.run(
        app,
        host=LISTEN_HOST,
        port=chosen_port,
        log_level=LOG_LEVEL.lower(),
        loop="uvloop" if "uvloop" in sys.modules else "asyncio",
        http="httptools",
        timeout_keep_alive=int(KEEPALIVE_EXPIRY),
        access_log=False,
    )


if __name__ == "__main__":
    main()
