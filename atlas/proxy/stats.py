"""
Persistent, thread-safe proxy statistics tracker.

Records per-request token/tool/model usage, persists to disk, and derives
average tokens-per-request for capacity estimation.

Atlas is a single-provider (NVIDIA) proxy, so the per-provider split is
collapsed to a single counter. The `provider` argument is kept on the
record_* signatures for call-site compatibility but is not used to branch
counters.

Schema (JSON on disk):
{
  "started_at": "2026-07-11T00:00:00Z",
  "restart": {
    "requests": 0, "successes": 0, "failures": 0,
    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    "tool_calls": 0,
    "provider_nvidia": 0,
    "models": {}
  },
  "all_time": {
    "requests": 0, "successes": 0, "failures": 0,
    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    "tool_calls": 0,
    "provider_nvidia": 0,
    "models": {}
  }
}
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATS_DIR = Path(os.environ.get("ATLAS_STATS_DIR", str(Path(__file__).resolve().parents[1] / "data")))
STATS_FILE = STATS_DIR / "proxy_stats.json"
_STATS_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_bucket() -> dict[str, Any]:
    return {
        "requests": 0,
        "successes": 0,
        "failures": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "tool_calls": 0,
        "provider_nvidia": 0,
        "models": {},
    }


def load() -> dict[str, Any]:
    """Load stats from disk. Returns fresh data if file missing/corrupt."""
    try:
        raw = STATS_FILE.read_text()
        data = json.loads(raw)
        # Ensure schema completeness
        if "started_at" not in data:
            data["started_at"] = _now_iso()
        for section in ("restart", "all_time"):
            if section not in data:
                data[section] = _empty_bucket()
            for k, v in _empty_bucket().items():
                if k not in data[section]:
                    data[section][k] = v if not isinstance(v, dict) else {}
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"started_at": _now_iso(), "restart": _empty_bucket(), "all_time": _empty_bucket()}


def save(data: dict[str, Any]) -> None:
    """Atomically write stats to disk."""
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATS_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(STATS_FILE))
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _merge_bucket(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Merge src bucket into dst bucket (all_time accumulates restart)."""
    dst["requests"] += src["requests"]
    dst["successes"] += src["successes"]
    dst["failures"] += src["failures"]
    dst["prompt_tokens"] += src["prompt_tokens"]
    dst["completion_tokens"] += src["completion_tokens"]
    dst["total_tokens"] += src["total_tokens"]
    dst["tool_calls"] += src["tool_calls"]
    dst["provider_nvidia"] += src["provider_nvidia"]
    for model, count in src.get("models", {}).items():
        dst["models"][model] = dst["models"].get(model, 0) + count


def record_success(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    tool_calls: int,
) -> None:
    """Record a successful request. `provider` kept for call-site compatibility."""
    with _STATS_LOCK:
        data = load()
        bucket = data["restart"]
        bucket["requests"] += 1
        bucket["successes"] += 1
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["total_tokens"] += total_tokens
        bucket["tool_calls"] += tool_calls
        bucket["provider_nvidia"] += 1
        bucket["models"][model] = bucket["models"].get(model, 0) + 1
        _merge_bucket(data["all_time"], bucket)
        save(data)


def record_failure(provider: str) -> None:
    """Record a failed request. `provider` kept for call-site compatibility."""
    with _STATS_LOCK:
        data = load()
        bucket = data["restart"]
        bucket["requests"] += 1
        bucket["failures"] += 1
        bucket["provider_nvidia"] += 1
        _merge_bucket(data["all_time"], bucket)
        save(data)


def reset_since_restart() -> dict[str, Any]:
    """Clear restart stats, return old restart data."""
    with _STATS_LOCK:
        data = load()
        old = data["restart"]
        data["restart"] = _empty_bucket()
        save(data)
        return old


def get_status() -> dict[str, Any]:
    """Return full stats snapshot (all-time + restart + estimation)."""
    with _STATS_LOCK:
        data = load()
        rt = data["restart"]
        at = data["all_time"]

        def _avg_tokens(bucket: dict[str, Any]) -> float:
            if bucket["requests"] == 0:
                return 0.0
            return bucket["total_tokens"] / bucket["requests"]

        avg_tokens_rt = _avg_tokens(rt)
        avg_tokens_at = _avg_tokens(at)
        # Use all-time average for capacity estimation (more data = more accurate)
        avg_tokens = avg_tokens_at if avg_tokens_at > 0 else avg_tokens_rt

        uptime = time.time() - _parse_iso(data.get("started_at", ""))

        return {
            "started_at": data.get("started_at", ""),
            "restart": {
                **rt,
                "avg_tokens_per_request": round(avg_tokens_rt, 1),
            },
            "all_time": {
                **at,
                "avg_tokens_per_request": round(avg_tokens_at, 1),
            },
            "uptime_seconds": round(uptime, 0),
        }


def _parse_iso(iso: str) -> float:
    """Parse ISO timestamp to unix epoch, return 0 on failure."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0
