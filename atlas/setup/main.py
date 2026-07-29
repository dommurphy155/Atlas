"""Atlas Setup — Main entry point."""

import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .modules.console import get_console, AtlasConsole
from .modules.platform import PlatformInfo, detect_platform, is_root
from .modules.files import ensure_dir
from .modules.env import EnvConfig, ensure_env_file, install_shell_env, install_shell_env_windows
from .modules.keys import collect_keys, resolve_keys_file
from .modules.service import ServiceConfig, install_service
from .modules.cli import install_cli
from .modules.health import check_proxy_health


console = get_console()


@dataclass
class SetupContext:
    """All state for the setup run."""
    platform: PlatformInfo
    provider: str = "openrouter"
    non_interactive: bool = False
    dry_run: bool = False
    skip_keys: bool = False
    skip_service: bool = False
    skip_cli: bool = False
    skip_env: bool = False


def print_banner(console: AtlasConsole) -> None:
    """Print the Atlas banner."""
    console.banner(
        "Atlas Proxy Setup v2",
        "Port 8788 · OpenRouter/NVIDIA · Claude Code front-end"
    )


def check_prerequisites(ctx: SetupContext) -> bool:
    """Check system prerequisites."""
    console.step("Checking prerequisites")

    # Python version
    if sys.version_info < (3, 10):
        console.error("Python 3.10+ required")
        return False
    console.success(f"Python {sys.version_info.major}.{sys.version_info.minor}")

    # Project structure
    if not (ctx.platform.project_dir / "proxy" / "main.py").exists():
        console.error("Not in Atlas project directory (missing proxy/main.py)")
        return False
    console.success("Project structure OK")

    # Venv
    venv_python = ctx.platform.venv_bin / ("python.exe" if ctx.platform.is_windows else "python")
    if not venv_python.exists():
        console.error(f"Virtualenv not found at {ctx.platform.venv_dir}")
        console.info("Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt")
        return False
    console.success(f"Venv: {venv_python}")

    # Check proxy deps
    try:
        subprocess.run(
            [str(venv_python), "-c", "import fastapi, uvicorn, httpx, orjson"],
            check=True,
            capture_output=True,
        )
        console.success("Proxy dependencies installed")
    except subprocess.CalledProcessError:
        console.error("Missing proxy dependencies")
        console.info("Run: .venv/bin/pip install fastapi uvicorn httpx orjson uvloop")
        return False

    return True


def configure_provider(ctx: SetupContext) -> str:
    """Select provider."""
    console.step("Provider selection")

    if ctx.non_interactive:
        console.info(f"Using provider: {ctx.provider}")
        return ctx.provider

    console.info("Available providers:")
    console.info("  1) OpenRouter (default) — 600+ free models via OpenRouter")
    console.info("  2) NVIDIA — Nemotron 3 Ultra via NVIDIA API")

    choice = console.prompt("Select provider [1/2]", default="1")
    provider = "openrouter" if choice in ("1", "", "openrouter") else "nvidia"
    console.success(f"Selected: {provider}")
    return provider


def collect_api_keys(ctx: SetupContext, provider: str) -> list[str]:
    """Collect API keys for the selected provider."""
    if ctx.skip_keys:
        console.step("Skipping key collection (--skip-keys)")
        return []

    console.step(f"Collecting {provider.title()} API keys")

    # Resolve keys file
    env_var = "ATLAS_OPENROUTER_KEYS_FILE" if provider == "openrouter" else "ATLAS_NVIDIA_KEYS_FILE"
    keys_file = resolve_keys_file(provider, ctx.platform.project_dir, os.environ.get(env_var))

    # Load existing
    from .modules.keys import read_keys, validate_key
    all_keys = read_keys(keys_file)
    validator = lambda k: validate_key(k, provider)
    existing = [k for k in all_keys if validator(k)]

    min_keys = 6 if provider == "openrouter" else 6

    if len(existing) >= min_keys and not ctx.non_interactive:
        console.success(f"Found {len(existing)} valid keys (minimum {min_keys})")
        if not console.confirm("Re-enter keys anyway?", default=False):
            return existing

    if ctx.non_interactive:
        if len(existing) < min_keys:
            console.warning(f"Only {len(existing)} keys, need {min_keys}. Non-interactive mode: continuing.")
        return existing

    # Interactive collection
    keys = collect_keys(provider, keys_file, min_keys, existing)
    return keys


def setup_environment(ctx: SetupContext, provider: str, keys: list[str]) -> EnvConfig:
    """Configure .env and shell environment."""
    if ctx.skip_env:
        console.step("Skipping environment setup (--skip-env)")
        return EnvConfig()

    console.step("Configuring environment")

    config = EnvConfig()
    config.provider = provider
    config.proxy_host = "127.0.0.1"
    config.proxy_port = 8788

    # Keys file paths
    data_dir = ctx.platform.project_dir / "data"
    if provider == "openrouter":
        config.openrouter_keys_file = str(data_dir / "openroute_keys.txt")
    else:
        config.nvidia_keys_file = str(data_dir / "keys.txt")

    # System prompt override
    config.system_prompt_override_file = str(ctx.platform.project_dir / "data" / "system_prompt_override.txt")

    # Write .env
    ensure_env_file(ctx.platform, config)

    # Shell exports
    if ctx.platform.is_windows:
        install_shell_env_windows(config)
    else:
        install_shell_env(ctx.platform, config)

    console.success("Environment configured")
    return config


def setup_service(ctx: SetupContext, config: EnvConfig) -> bool:
    """Install and start system service."""
    if ctx.skip_service:
        console.step("Skipping service install (--skip-service)")
        return True

    console.step("Installing system service")

    svc_config = ServiceConfig(
        name="atlas-proxy",
        display_name="Atlas Proxy",
        description="Atlas Multi-Provider Proxy (OpenRouter/NVIDIA)",
        working_dir=str(ctx.platform.project_dir),
        python_path=str(ctx.platform.venv_bin / ("python.exe" if ctx.platform.is_windows else "python")),
        module="proxy.main",
        user=os.getenv("SUDO_USER") or os.getenv("USER") or "root",
        env=config.to_env_dict(),
    )

    if ctx.dry_run:
        console.info("Dry run: would install service")
        return True

    return install_service(svc_config, ctx.platform)


def setup_cli(ctx: SetupContext) -> bool:
    """Install CLI binary."""
    if ctx.skip_cli:
        console.step("Skipping CLI install (--skip-cli)")
        return True

    console.step("Installing CLI")
    return install_cli(ctx.platform, ctx.platform.project_dir)


def run_health_check(ctx: SetupContext) -> bool:
    """Verify proxy is healthy."""
    console.step("Health check")

    if ctx.dry_run:
        console.info("Dry run: skipping health check")
        return True

    result = check_proxy_health()
    if result.healthy:
        console.success(f"Proxy healthy — {result.stats.get('total_keys', 0)} keys loaded")
        return True
    else:
        console.error(f"Health check failed: {result.error}")
        return False


def launch_claude(ctx: SetupContext) -> None:
    """Launch Claude Code."""
    console.step("Launching Claude Code")

    # Check if claude is available
    claude = ctx.platform.venv_python.parent / "claude"
    if not claude.exists():
        # Check system PATH
        import shutil
        claude = shutil.which("claude")

    if claude:
        console.success(f"Found claude: {claude}")
        if not ctx.non_interactive and console.confirm("Launch Claude Code now?", default=True):
            os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:8788"
            os.environ["ANTHROPIC_API_KEY"] = "atlas"
            os.execvp(str(claude), ["claude"])
    else:
        console.warning("Claude Code not installed")
        console.info("Install: curl -fsSL https://claude.ai/install.sh | bash")


def run_setup(
    provider: str = "openrouter",
    non_interactive: bool = False,
    dry_run: bool = False,
    skip_keys: bool = False,
    skip_service: bool = False,
    skip_cli: bool = False,
    skip_env: bool = False,
) -> int:
    """Main setup orchestration."""
    ctx = SetupContext(
        platform=detect_platform(),
        provider=provider,
        non_interactive=non_interactive,
        dry_run=dry_run,
        skip_keys=skip_keys,
        skip_service=skip_service,
        skip_cli=skip_cli,
        skip_env=skip_env,
    )

    print_banner(console)

    # Show context
    console.table("Setup Context", ["Field", "Value"], [
        ["Platform", f"{ctx.platform.system} ({ctx.platform.shell})"],
        ["Provider", ctx.provider],
        ["Project", str(ctx.platform.project_dir)],
        ["Venv", str(ctx.platform.venv_bin / ("python.exe" if ctx.platform.is_windows else "python"))],
        ["Non-interactive", str(ctx.non_interactive)],
        ["Dry run", str(ctx.dry_run)],
    ])

    if ctx.dry_run:
        console.warning("DRY RUN — no changes will be made")

    if not ctx.non_interactive and not ctx.dry_run:
        if not console.confirm("Continue?", default=True):
            console.info("Aborted")
            return 0

    # Prerequisites
    if not check_prerequisites(ctx):
        return 1

    # Provider selection
    if not ctx.non_interactive and not ctx.dry_run:
        ctx.provider = configure_provider(ctx)

    # API Keys
    keys = collect_api_keys(ctx, ctx.provider)

    # Environment
    config = setup_environment(ctx, ctx.provider, keys)

    # Service
    if not setup_service(ctx, config):
        return 1

    # CLI
    if not setup_cli(ctx):
        return 1

    # Health check
    if not run_health_check(ctx):
        console.error("Setup completed but proxy is not healthy")
        console.info("Check logs: sudo journalctl -u atlas-proxy -f")
        return 1

    # Success summary
    console.panel(
        f"[bold green]Atlas Proxy Setup Complete![/]\n\n"
        f"Provider: [cyan]{ctx.provider}[/]\n"
        f"Proxy: [cyan]http://127.0.0.1:8788[/]\n"
        f"Keys: [cyan]{len(keys)}[/] loaded\n"
        f"Service: [cyan]atlas-proxy[/] (systemd)\n"
        f"CLI: [cyan]atlas[/] — try 'atlas status'",
        title="Summary",
        style="green"
    )

    # Launch Claude
    if not ctx.non_interactive and not ctx.dry_run:
        launch_claude(ctx)

    return 0


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="atlas-setup",
        description="Atlas Proxy v2 — Cross-platform installer",
    )
    parser.add_argument("--provider", choices=["openrouter", "nvidia"], default="openrouter")
    parser.add_argument("--non-interactive", action="store_true", help="No prompts, use defaults")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--skip-keys", action="store_true", help="Skip API key collection")
    parser.add_argument("--skip-service", action="store_true", help="Skip systemd/launchd install")
    parser.add_argument("--skip-cli", action="store_true", help="Skip CLI binary install")
    parser.add_argument("--skip-env", action="store_true", help="Skip .env and shell config")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    # Recreate console with flags
    global console
    console = get_console(quiet=args.quiet, json_output=args.json)

    try:
        return run_setup(
            provider=args.provider,
            non_interactive=args.non_interactive,
            dry_run=args.dry_run,
            skip_keys=args.skip_keys,
            skip_service=args.skip_service,
            skip_cli=args.skip_cli,
            skip_env=args.skip_env,
        )
    except KeyboardInterrupt:
        console.error("\nInterrupted")
        return 130
    except Exception as e:
        console.error(f"Setup failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())