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
    FORCE_DEFAULT_MODEL,
    KEEPALIVE_EXPIRY,
    LISTEN_HOST,
    LISTEN_PORT,
    LOG_LEVEL,
    get_default_model,
    get_fallback_keys_file,
    get_keys_file,
    get_provider,
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
    provider = get_provider()
    keys = load_keys(get_keys_file())
    if not keys:
        keys = load_keys(get_fallback_keys_file())
        if keys:
            log.warning(
                "Primary keys file missing – using fallback %s (%d keys)",
                get_fallback_keys_file(),
                len(keys),
            )
    if not keys:
        log.error(
            "No keys found. Expected file: %s (one key per line)",
            get_keys_file(),
        )
        sys.exit(1)

    log.info("Loaded %d %s keys", len(keys), provider)
    pool = KeyPool(keys)
    core = ProxyCore(pool)
    await core.start()
    routes.proxy = core
    log.info(
        "Proxy listening on http://%s:%d  (keys=%d, healthy=%d, "
        "provider=%s, default_model=%s, force=%s)",
        LISTEN_HOST,
        LISTEN_PORT,
        pool.stats()["total"],
        pool.stats()["healthy"],
        provider,
        get_default_model(),
        FORCE_DEFAULT_MODEL,
    )
    yield
    await core.stop()
    routes.proxy = None
    log.info("Shutdown complete")


app = FastAPI(
    title="Atlas Translation Proxy",
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
