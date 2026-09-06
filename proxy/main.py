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
        LISTEN_PORT,
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


def _pick_port(preferred: int) -> int:
    """Return ``preferred`` if it's free; otherwise pick a random free
    TCP port in the high range. NEVER touches the existing occupant —
    we don't own whatever process is on ``preferred`` so we just yield.
    """
    import socket

    def _free(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return True
            except OSError:
                return False

    if _free(preferred):
        return preferred

    # Try a handful of random high ports. Using random makes us unlikely
    # to collide with other atlas forks doing the same dance on the same
    # host, but the loop guarantees we land on something free.
    import random
    for _ in range(50):
        candidate = random.randint(49152, 65535)
        if _free(candidate):
            return candidate

    # Last resort: ask the OS for any free port (kernel-assigned).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def main() -> None:
    import uvicorn

    # If our preferred port is occupied by something else (e.g. another
    # atlas install, prod, or a stale dev process), pick a random free
    # one instead of killing the occupant. The chosen port is written
    # to data/bound_port so the installer's health check and `atlas
    # status` know where to find us, since LISTEN_PORT in config may
    # not match the actual bind port.
    chosen_port = _pick_port(LISTEN_PORT)
    if chosen_port != LISTEN_PORT:
        import logging
        logging.getLogger("proxy").warning(
            "LISTEN_PORT %d is occupied; falling back to %d (occupant untouched)",
            LISTEN_PORT, chosen_port,
        )
    try:
        from pathlib import Path as _P
        _bound = _P(__file__).resolve().parent.parent / "data" / "bound_port"
        _bound.parent.mkdir(parents=True, exist_ok=True)
        _bound.write_text(str(chosen_port))
    except OSError:
        pass

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
