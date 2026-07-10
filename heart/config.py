"""Central configuration. One place for every knob, loaded from env at import."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; env vars still work without it
    def load_dotenv(_path: Path) -> None:
        return None

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0") == "1"


# ── Server ──────────────────────────────────────────────────────────────
HOST: str = _env("NVDA_PROXY_HOST", "127.0.0.1")
PORT: int = _env_int("NVDA_PROXY_PORT", 8787)
DEBUG: bool = _env_bool("NVDA_PROXY_DEBUG", False)
MAX_BODY_BYTES: int = 2 * 1024 * 1024  # 2 MiB request cap

# ── Keys ────────────────────────────────────────────────────────────────
KEYS_FILE: str = _env("NVDA_KEYS_FILE", str(ROOT_DIR / "data" / "keys.txt"))
RELOAD_SECONDS: int = _env_int("NVDA_PROXY_RELOAD_SECONDS", 5)

# ── Upstream (NVIDIA) ───────────────────────────────────────────────────
NVIDIA_BASE_URL: str = _env(
    "NVDA_PROXY_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
DEFAULT_MODEL: str = _env("NVDA_PROXY_MODEL", "z-ai/glm-5.2")
REQUEST_TIMEOUT: float = _env_float("NVDA_PROXY_REQUEST_TIMEOUT", 300.0)

# ── Retry / failover ────────────────────────────────────────────────────
MAX_RETRIES: int = _env_int("NVDA_PROXY_MAX_RETRIES", 2)          # 5xx retries
MAX_FAILOVERS: int = _env_int("NVDA_PROXY_MAX_FAILOVERS", 3)      # distinct keys per request
COOLDOWN_SECONDS: int = _env_int("NVDA_PROXY_COOLDOWN_SECONDS", 60)  # default 429 cooldown

# ── Stats ───────────────────────────────────────────────────────────────
STATS_DIR: str = _env("NVDA_STATS_DIR", str(ROOT_DIR / "data"))
