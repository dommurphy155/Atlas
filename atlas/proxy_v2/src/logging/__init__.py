"""Logging and metrics."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RequestLog:
    """Request log entry."""
    rid: str
    method: str
    path: str
    status: int
    duration: float
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None


class Logger:
    """Structured logger."""

    def __init__(self, name: str = "proxy"):
        self.logger = logging.getLogger(name)
        self._setup()

    def _setup(self):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra=kwargs)

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, extra=kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra=kwargs)


class Stats:
    """Statistics tracker."""

    def __init__(self):
        self.requests = 0
        self.successes = 0
        self.failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def record(self, success: bool, prompt: int = 0, completion: int = 0):
        self.requests += 1
        if success:
            self.successes += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
        else:
            self.failures += 1

    def get(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


# Global instances
logger = Logger()
stats = Stats()
