"""Structured logging with request-ID binding.

On import: configures root logger, exposes `get_logger()`, `log`, and request-ID helpers.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Config from env (read once at import)
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_JSON: bool = os.environ.get("LOG_JSON", "").strip().lower() in ("1", "true", "yes", "on")
LOG_REQUEST_ID: bool = os.environ.get("LOG_REQUEST_ID", "").strip().lower() in ("1", "true", "yes", "on")

# ---------------------------------------------------------------------------
# Request-ID thread-local storage
# ---------------------------------------------------------------------------
_request_id_local = threading.local()


def _get_request_id() -> Optional[str]:
    return getattr(_request_id_local, "value", None)


def set_request_id(rid: Optional[str]) -> None:
    """Set request ID for current thread (call at start of request)."""
    if rid:
        _request_id_local.value = rid
    elif hasattr(_request_id_local, "value"):
        del _request_id_local.value


def clear_request_id() -> None:
    """Clear request ID for current thread (call at end of request)."""
    if hasattr(_request_id_local, "value"):
        del _request_id_local.value


@contextmanager
def bind_request_id(rid: str):
    """Context manager to bind request ID for the duration of a block."""
    set_request_id(rid)
    try:
        yield
    finally:
        clear_request_id()


# ---------------------------------------------------------------------------
# JSON formatter (optional)
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter with optional request_id."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import time

        data: Dict[str, Any] = {
            "ts": time.strftime("%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if LOG_REQUEST_ID:
            rid = _get_request_id()
            if rid:
                data["request_id"] = rid
        # Include extra fields
        for k, v in record.__dict__.items():
            if k not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "exc_info",
                "exc_text",
                "stack_info",
            }:
                data[k] = v
        return json.dumps(data, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Root logger setup (runs on import)
# ---------------------------------------------------------------------------
_handler = logging.StreamHandler(sys.stdout)
if LOG_JSON:
    _handler.setFormatter(JsonFormatter())
else:
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )

_root = logging.getLogger()
_root.handlers.clear()
_root.addHandler(_handler)
_root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Default logger for this package
log = logging.getLogger("or-proxy")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Get a logger named `or-proxy.<name>`."""
    return logging.getLogger(f"or-proxy.{name}")


# Re-export request-id helpers
__all__ = [
    "LOG_LEVEL",
    "LOG_JSON",
    "LOG_REQUEST_ID",
    "log",
    "get_logger",
    "set_request_id",
    "clear_request_id",
    "bind_request_id",
]