"""Structured logging for the Atlas proxy.

On import: configures the root logger and exposes `get_logger()` and `log`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Config from env (read once at import)
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_JSON: bool = os.environ.get("LOG_JSON", "").strip().lower() in ("1", "true", "yes", "on")

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
log = logging.getLogger("atlas-proxy")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Get a logger named `atlas-proxy.<name>`."""
    return logging.getLogger(f"atlas-proxy.{name}")


# Re-export request-id helpers
__all__ = [
    "LOG_LEVEL",
    "LOG_JSON",
    "log",
    "get_logger",
]