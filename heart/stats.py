"""Persistent request stats. Restart + all-time buckets, atomic writes."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

STATS_FILE = Path(config.STATS_DIR) / "proxy_stats.json"
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
        "models": {},
    }


def load() -> dict[str, Any]:
    try:
        raw = STATS_FILE.read_text()
        data = json.loads(raw)
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
        return {
            "started_at": _now_iso(),
            "restart": _empty_bucket(),
            "all_time": _empty_bucket(),
        }


def save(data: dict[str, Any]) -> None:
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATS_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(STATS_FILE))
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k in (
        "requests", "successes", "failures",
        "prompt_tokens", "completion_tokens", "total_tokens", "tool_calls",
    ):
        dst[k] += src[k]
    for model, count in src.get("models", {}).items():
        dst["models"][model] = dst["models"].get(model, 0) + count


def record_success(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    tool_calls: int,
) -> None:
    with _STATS_LOCK:
        data = load()
        bucket = data["restart"]
        bucket["requests"] += 1
        bucket["successes"] += 1
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["total_tokens"] += total_tokens
        bucket["tool_calls"] += tool_calls
        bucket["models"][model] = bucket["models"].get(model, 0) + 1
        _merge(data["all_time"], bucket)
        save(data)


def record_failure() -> None:
    with _STATS_LOCK:
        data = load()
        bucket = data["restart"]
        bucket["requests"] += 1
        bucket["failures"] += 1
        _merge(data["all_time"], bucket)
        save(data)


def get_status() -> dict[str, Any]:
    with _STATS_LOCK:
        data = load()
        rt = data["restart"]
        at = data["all_time"]

        def avg(b: dict[str, Any]) -> float:
            return round(b["total_tokens"] / b["requests"], 1) if b["requests"] else 0.0

        uptime = time.time() - _parse_iso(data.get("started_at", ""))
        return {
            "started_at": data.get("started_at", ""),
            "restart": {**rt, "avg_tokens_per_request": avg(rt)},
            "all_time": {**at, "avg_tokens_per_request": avg(at)},
            "uptime_seconds": round(uptime, 0),
        }


def _parse_iso(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0
