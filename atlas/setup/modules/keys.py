"""API key collection and management."""

import os
import re
from pathlib import Path
from typing import Optional

from .console import get_console
from .files import read_file, write_file
from .platform import detect_platform

console = get_console()


# Key validation patterns
KEY_PATTERNS = {
    "openrouter": re.compile(r"^sk-or-v1-[a-zA-Z0-9_-]+$"),
    "nvidia": re.compile(r"^nvapi-[a-zA-Z0-9_-]+$"),
}


def validate_key(key: str, provider: str) -> bool:
    """Validate API key format."""
    pattern = KEY_PATTERNS.get(provider.lower())
    if not pattern:
        return bool(key and len(key) > 10)
    return bool(pattern.match(key))


def read_keys(path: Path) -> list[str]:
    """Read keys from file, one per line."""
    content = read_file(path)
    if not content:
        return []

    keys = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Support KEY=value format
        if "=" in line and not line.startswith(("sk-", "nvapi-")):
            line = line.split("=", 1)[1].strip()
        if line:
            keys.append(line)
    return keys


def write_keys(path: Path, keys: list[str]) -> bool:
    """Write keys to file, one per line, with 0600 permissions."""
    content = "\n".join(keys) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)
        return True
    except Exception as e:
        console.error(f"Failed to write keys: {e}")
        return False


def collect_keys(
    provider: str,
    keys_file: Path,
    min_keys: int = 6,
    existing: Optional[list[str]] = None,
) -> list[str]:
    """Interactively collect API keys."""
    existing = existing or []
    seen = set(existing)

    # Filter valid existing keys
    valid_existing = [k for k in existing if validate_key(k, provider)]
    if len(valid_existing) != len(existing):
        console.warning(f"Filtered {len(existing) - len(valid_existing)} invalid existing keys")

    console.step(f"Collecting {provider.title()} API keys")
    console.info(f"Keys file: {keys_file}")
    console.info(f"Valid existing keys: {len(valid_existing)}")

    if len(valid_existing) >= min_keys:
        console.success(f"Already have {len(valid_existing)} keys (minimum {min_keys})")
        # Write cleaned list
        write_keys(keys_file, valid_existing)
        return valid_existing

    needed = min_keys - len(valid_existing)
    console.info(f"Need {needed} more key(s) to reach minimum of {min_keys}")

    console.info(f"\n[bold]Where to get {provider.title()} keys:[/]")
    if provider == "openrouter":
        console.info("  • OpenRouter: https://openrouter.ai/keys")
        console.info("  • Use throwaway emails: https://temp-mail.org / https://www.agentmail.to")
    elif provider == "nvidia":
        console.info("  • NVIDIA API: https://build.nvidia.com/explore/discover")
        console.info("  • Each key needs separate account")

    console.info("\nPaste one key per line. Empty line or 'done' to finish.")

    collected = list(valid_existing)
    while len(collected) < min_keys:
        remaining = min_keys - len(collected)
        prompt = f"  Key {len(collected) + 1}"
        if remaining > 0:
            prompt += f" ({remaining} more needed)"
        prompt += " > "

        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            console.warning("\nInput interrupted")
            break

        if not raw or raw.lower() in ("done", "quit", "exit"):
            if len(collected) < min_keys:
                console.warning(f"Only {len(collected)}/{min_keys} keys collected")
            break

        if not validate_key(raw, provider):
            console.error(f"Invalid {provider} key format")
            console.info(f"  Expected: {KEY_PATTERNS[provider].pattern if provider in KEY_PATTERNS else 'sk-... or nvapi-...'}")
            continue

        if raw in seen:
            console.warning("Duplicate key, skipping")
            continue

        seen.add(raw)
        collected.append(raw)
        console.success(f"Accepted ({len(collected)} total)")

    # Write final list
    write_keys(keys_file, collected)
    console.success(f"Saved {len(collected)} keys to {keys_file}")
    return collected


def resolve_keys_file(
    provider: str,
    project_dir: Path,
    env_override: Optional[str] = None,
) -> Path:
    """Resolve keys file path with fallback."""
    # 1. Explicit env override
    if env_override:
        return Path(env_override)

    # 2. Provider-specific default
    if provider == "openrouter":
        default = project_dir / "data" / "openroute_keys.txt"
    else:
        default = project_dir / "data" / "keys.txt"

    # 3. Fallback to legacy
    legacy = project_dir / "data" / "keys.txt"
    if legacy.exists() and not default.exists():
        return legacy

    return default