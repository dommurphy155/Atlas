"""FastAPI app, lifespan, and uvicorn entrypoint.

Run:
  python -m proxy.main
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import routes
from .config import (
    CORS_ORIGINS,
    FALLBACK_KEY_FILE,
    FORCE_DEFAULT_MODEL,
    KEEPALIVE_EXPIRY,
    KEY_FILE,
    LISTEN_HOST,
    LISTEN_PORT,
    LOG_LEVEL,
    OPENROUTER_MODEL,
    log,
)
from .keypool import KeyPool, load_keys
from .proxy import ProxyCore

# Optional high-performance event loop
try:
    import uvloop

    uvloop.install()
except ImportError:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    keys = load_keys(KEY_FILE)
    if not keys:
        keys = load_keys(FALLBACK_KEY_FILE)
        if keys:
            log.warning(
                "Primary keys file missing – using fallback %s (%d keys)",
                FALLBACK_KEY_FILE,
                len(keys),
            )
    if not keys:
        log.error(
            "No keys found. Expected file: %s (one sk-or-… key per line)",
            KEY_FILE,
        )
        sys.exit(1)

    log.info("Loaded %d OpenRouter keys", len(keys))
    pool = KeyPool(keys)
    core = ProxyCore(pool)
    await core.start()
    routes.proxy = core
    log.info(
        "Proxy listening on http://%s:%d  (keys=%d, healthy=%d, "
        "default_model=%s, force=%s)",
        LISTEN_HOST,
        LISTEN_PORT,
        pool.stats()["total"],
        pool.stats()["healthy"],
        OPENROUTER_MODEL,
        FORCE_DEFAULT_MODEL,
    )
    yield
    await core.stop()
    routes.proxy = None
    log.info("Shutdown complete")


app = FastAPI(
    title="OpenRouter Translation Proxy",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-request-id"],
)

app.include_router(routes.router)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "proxy.main:app",
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        log_level=LOG_LEVEL.lower(),
        loop="uvloop" if "uvloop" in sys.modules else "asyncio",
        http="httptools",
        timeout_keep_alive=int(KEEPALIVE_EXPIRY),
        access_log=False,
    )


if __name__ == "__main__":
    main()
