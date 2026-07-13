"""Token tracker — clean since-restart token/usage summary.

Reads the persisted stats file (the same one proxy.stats writes after every
request) and prints a one-glance summary of usage since the last proxy
restart: requests, successes, failures, in/out/total tokens, tool calls,
and per-model breakdown.

Invoked by the `atlas tokens` CLI command. Pure read — never mutates stats.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

STATS_FILE = Path(
    os.environ.get(
        "ATLAS_STATS_FILE",
        str(Path(__file__).resolve().parents[1] / "data" / "proxy_stats.json"),
    )
)


def _load() -> dict[str, Any] | None:
    """Load the stats file. Returns None if missing or corrupt."""
    try:
        return json.loads(STATS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _fmt_int(n: int) -> str:
    """Group thousands with a comma."""
    return f"{n:,}"


def _bar(value: int, total: int, width: int = 20) -> str:
    """Tiny ASCII bar for the in/out split. Empty if total is zero."""
    if total <= 0:
        return ""
    filled = round((value / total) * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def render() -> str:
    """Render the since-restart token summary as a string."""
    data = _load()
    if data is None:
        return (
            f"no stats found at {STATS_FILE}\n"
            "is the proxy running? stats are written after the first request."
        )

    restart = data.get("restart") or {}
    started_at = data.get("started_at") or "?"

    requests = int(restart.get("requests") or 0)
    successes = int(restart.get("successes") or 0)
    failures = int(restart.get("failures") or 0)
    in_tokens = int(restart.get("prompt_tokens") or 0)
    out_tokens = int(restart.get("completion_tokens") or 0)
    total_tokens = int(restart.get("total_tokens") or 0)
    tool_calls = int(restart.get("tool_calls") or 0)
    models = restart.get("models") or {}

    success_rate = (successes / requests * 100) if requests else 0.0
    avg_tokens = (total_tokens / requests) if requests else 0.0

    lines: list[str] = []
    lines.append("atlas tokens — since restart")
    lines.append(f"started: {started_at}")
    lines.append("")
    lines.append(f"  requests      {_fmt_int(requests)}")
    lines.append(f"  successes     {_fmt_int(successes)}   ({success_rate:.1f}%)")
    lines.append(f"  failures      {_fmt_int(failures)}")
    lines.append("")
    lines.append("  tokens")
    lines.append(f"    in          {_fmt_int(in_tokens)}")
    lines.append(f"    out         {_fmt_int(out_tokens)}")
    lines.append(f"    total       {_fmt_int(total_tokens)}")
    if total_tokens:
        lines.append(f"    split       {_bar(in_tokens, total_tokens)} in/out")
    lines.append(f"    tool calls  {_fmt_int(tool_calls)}")
    lines.append("")
    lines.append(f"  avg tokens/req   {avg_tokens:.1f}")
    if models:
        lines.append("  models")
        for model, count in sorted(models.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"    {model:<28} {_fmt_int(int(count))}")
    return "\n".join(lines)


def main() -> None:
    print(render())


if __name__ == "__main__":
    main()
