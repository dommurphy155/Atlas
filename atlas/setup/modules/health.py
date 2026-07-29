"""Health checks and verification."""

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .console import get_console
from .platform import PlatformInfo, detect_platform

console = get_console()


@dataclass
class HealthResult:
    """Health check result."""
    healthy: bool
    stats: Optional[dict] = None
    error: Optional[str] = None


def check_proxy_health(
    url: str = "http://127.0.0.1:8788",
    timeout: float = 3.0,
    retries: int = 30,
    interval: float = 0.5,
) -> HealthResult:
    """Poll proxy /health until healthy or timeout."""
    health_url = f"{url}/health"
    keys_url = f"{url}/health/keys"

    console.info(f"Waiting for proxy at {health_url}...")

    for i in range(retries):
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "atlas-setup/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 300:
                    data = json.loads(resp.read().decode())
                    console.success(f"Proxy healthy (attempt {i+1}/{retries})")
                    # Now get keys
                    try:
                        with urllib.request.urlopen(keys_url, timeout=timeout) as kres:
                            keys = json.loads(kres.read().decode())
                            return HealthResult(healthy=True, stats=keys)
                    except Exception:
                        return HealthResult(healthy=True, stats=data.get("keys", {}))
        except Exception:
            pass

        if i < retries - 1:
            time.sleep(interval)

    return HealthResult(healthy=False, error=f"Proxy did not become healthy after {retries * interval:.0f}s")


def check_service_status(service_name: str, platform_info: PlatformInfo) -> bool:
    """Check if system service is running."""
    try:
        if platform_info.is_linux:
            result = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True, text=True, check=False
            )
            return result.stdout.strip() == "active"
        elif platform_info.is_macos:
            label = service_name.replace("-", ".")
            result = subprocess.run(
                ["launchctl", "list", label],
                capture_output=True, text=True, check=False
            )
            return result.returncode == 0
        elif platform_info.is_windows:
            result = subprocess.run(
                ["sc", "query", service_name],
                capture_output=True, text=True, check=False
            )
            return "RUNNING" in result.stdout
    except Exception:
        pass
    return False


def verify_installation(
    platform_info: PlatformInfo,
    service_name: str = "atlas-proxy",
    proxy_url: str = "http://127.0.0.1:8788",
) -> dict:
    """Run full post-install verification."""
    results = {
        "service_running": False,
        "proxy_healthy": False,
        "keys_loaded": 0,
        "cli_installed": False,
        "env_configured": False,
    }

    # Service
    results["service_running"] = check_service_status(service_name, platform_info)
    if results["service_running"]:
        console.success(f"Service {service_name} is running")
    else:
        console.error(f"Service {service_name} is NOT running")

    # Proxy health
    health = check_proxy_health(proxy_url)
    results["proxy_healthy"] = health.healthy
    if health.healthy:
        console.success("Proxy health check passed")
        if health.stats:
            results["keys_loaded"] = health.stats.get("summary", {}).get("total", 0)
            console.info(f"Keys loaded: {results['keys_loaded']}")
    else:
        console.error(f"Proxy health check failed: {health.error}")

    # CLI
    cli_path = platform_info.bin_dir / ("atlas.cmd" if platform_info.is_windows else "atlas")
    results["cli_installed"] = cli_path.exists()
    if results["cli_installed"]:
        console.success(f"CLI installed at {cli_path}")
    else:
        console.warning(f"CLI not found at {cli_path}")

    # Env (ANTHROPIC_*)
    import os
    results["env_configured"] = bool(os.environ.get("ANTHROPIC_BASE_URL"))
    if results["env_configured"]:
        console.success("ANTHROPIC_* environment configured")
    else:
        console.warning("ANTHROPIC_* not in current environment (may need shell restart)")

    return results