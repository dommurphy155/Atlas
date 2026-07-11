#!/usr/bin/env python3
"""Atlas — Python installer.

Runs after install.sh has built the venv and pip-installed requirements.
Installs the systemd unit, symlinks the atlas CLI, writes default .env,
and optionally wires ANTHROPIC_* env vars into the user's shell rc.

Atlas is the NVIDIA-only proxy on port 8788. It is separate from any other
proxy project and does not reference them.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Resolve PROJECT_DIR as the parent of this file's directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_DIR / ".venv"
PYTHON_BIN = VENV_DIR / "bin" / "python"

UNIT_SRC = PROJECT_DIR / "systemd" / "atlas-proxy.service"
UNIT_DST = Path("/etc/systemd/system/atlas-proxy.service")
CLI_SRC = PROJECT_DIR / "bin" / "atlas"
CLI_DST = Path("/usr/local/bin/atlas")
ENV_FILE = PROJECT_DIR / ".env"

DEFAULT_ENV = """\
# Atlas — NVIDIA-only proxy environment defaults
ATLAS_PROXY_HOST=127.0.0.1
ATLAS_PROXY_PORT=8788
ATLAS_KEYS_FILE=/root/claude/atlas/data/keys.txt
ATLAS_NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1/chat/completions
ATLAS_NVIDIA_MODEL=z-ai/glm-5.2
ATLAS_PROXY_RELOAD_SECONDS=5
ATLAS_PROXY_MAX_KEY_FAILOVERS=3
ATLAS_PROXY_REQUEST_TIMEOUT=300
ATLAS_PROXY_MAX_RETRIES=2
ATLAS_PROXY_DEBUG=0
"""


def _run(cmd, **kwargs):
    """Run a command, streaming output to the console."""
    return subprocess.run(cmd, check=True, **kwargs)


def _sudo_run(cmd, **kwargs):
    """Run a command via sudo if we're not root, else directly."""
    if os.geteuid() != 0:
        cmd = ["sudo", *cmd]
    return _run(cmd, **kwargs)


def install_systemd_unit() -> None:
    """Install atlas-proxy.service into /etc/systemd/system and daemon-reload.

    Does NOT enable or start the unit — that is left to the operator via the
    `atlas` CLI. Only the single atlas-proxy.service unit is installed.
    """
    console.print(Panel.fit("[bold cyan]Installing systemd unit[/]", border_style="cyan"))

    if not UNIT_SRC.exists():
        console.print(f"[red]Unit source not found:[/red] {UNIT_SRC}")
        sys.exit(1)

    unit_text = UNIT_SRC.read_text()

    if shutil.which("systemctl") is None:
        console.print(
            "[yellow]systemctl not found on this system — skipping unit install.[/yellow]\n"
            "Install atlas-proxy.service into /etc/systemd/system manually if needed."
        )
        return

    # Write the unit file (via sudo tee if not root).
    if os.geteuid() == 0:
        UNIT_DST.write_text(unit_text)
    else:
        proc = subprocess.run(
            ["sudo", "tee", str(UNIT_DST)],
            input=unit_text,
            text=True,
            stdout=subprocess.DEVNULL,
            check=True,
        )

    console.print(f"[green]Wrote unit:[/green] {UNIT_DST}")
    _sudo_run(["systemctl", "daemon-reload"])
    console.print("[green]systemctl daemon-reload complete[/green]")
    console.print("[dim]Unit installed but NOT enabled/started. Use `atlas start` to run it.[/dim]")


def install_cli() -> None:
    """Symlink /usr/local/bin/atlas -> bin/atlas and make it executable."""
    console.print(Panel.fit("[bold cyan]Installing atlas CLI[/]", border_style="cyan"))

    if not CLI_SRC.exists():
        console.print(f"[red]CLI source not found:[/red] {CLI_SRC}")
        sys.exit(1)

    CLI_SRC.chmod(0o755)

    # Remove an existing symlink/path at the destination if present.
    if CLI_DST.is_symlink() or CLI_DST.exists():
        _sudo_run(["rm", "-f", str(CLI_DST)])

    _sudo_run(["ln", "-s", str(CLI_SRC), str(CLI_DST)])
    console.print(f"[green]Linked CLI:[/green] {CLI_DST} -> {CLI_SRC}")


def configure_env_defaults() -> None:
    """Create .env with ATLAS_ defaults if it does not already exist."""
    console.print(Panel.fit("[bold cyan]Configuring .env defaults[/]", border_style="cyan"))

    if ENV_FILE.exists():
        console.print(f"[yellow].env already exists — not overwriting:[/yellow] {ENV_FILE}")
        return

    ENV_FILE.write_text(DEFAULT_ENV)
    console.print(f"[green]Wrote default .env:[/green] {ENV_FILE}")


def wire_claude_env() -> None:
    """Optionally append ANTHROPIC_* exports to ~/.bashrc or ~/.zshrc.

    Only appends if the relevant rc file exists and the export lines are not
    already present. Uses port 8788 (the Atlas port).
    """
    console.print(Panel.fit("[bold cyan]Wiring shell env (optional)[/]", border_style="cyan"))

    candidates = [Path.home() / ".bashrc", Path.home() / ".zshrc"]
    rc_files = [p for p in candidates if p.exists()]

    if not rc_files:
        console.print("[yellow]No ~/.bashrc or ~/.zshrc found — skipping shell env wiring.[/yellow]")
        return

    block = (
        "\n# --- Atlas NVIDIA proxy (port 8788) ---\n"
        "export ANTHROPIC_BASE_URL=http://127.0.0.1:8788\n"
        "export ANTHROPIC_API_KEY=atlas\n"
        "# --- end Atlas ---\n"
    )

    for rc in rc_files:
        content = rc.read_text()
        if "ANTHROPIC_BASE_URL=http://127.0.0.1:8788" in content:
            console.print(f"[yellow]Already wired in:[/yellow] {rc}")
            continue

        with rc.open("a") as fh:
            fh.write(block)
        console.print(f"[green]Appended Atlas env exports to:[/green] {rc}")

    console.print("[dim]Restart your shell (or source the rc file) for the exports to take effect.[/dim]")


def main() -> None:
    console.print(
        Panel.fit(
            "[bold]Atlas NVIDIA Proxy — installer[/bold]\n"
            "Standalone NVIDIA-only proxy on port 8788",
            border_style="magenta",
        )
    )

    console.print(f"Project dir: [cyan]{PROJECT_DIR}[/]")
    console.print(f"Venv:        [cyan]{VENV_DIR}[/]")
    console.print(f"Python:      [cyan]{PYTHON_BIN}[/]")

    steps = [
        ("Install systemd unit", install_systemd_unit),
        ("Install atlas CLI", install_cli),
        ("Configure .env defaults", configure_env_defaults),
        ("Wire shell ANTHROPIC env", wire_claude_env),
    ]

    for label, fn in steps:
        console.print()
        fn()

    # Final summary.
    summary = Table(title="Atlas install summary", show_header=True, header_style="bold magenta")
    summary.add_column("Field", style="cyan", no_wrap=True)
    summary.add_column("Value", style="green")
    summary.add_row("Service", "atlas-proxy.service")
    summary.add_row("Port", "8788")
    summary.add_row("Unit file", str(UNIT_DST))
    summary.add_row("CLI", str(CLI_DST))
    summary.add_row("Env file", str(ENV_FILE))
    summary.add_row("Project", str(PROJECT_DIR))
    console.print()
    console.print(summary)

    console.print()
    console.print(
        Panel.fit(
            "[bold green]Atlas installed.[/bold green]\n"
            "Start it with:  [bold]atlas start[/]\n"
            "Status:         [bold]systemctl status atlas-proxy[/]\n"
            "Logs:           [bold]journalctl -u atlas-proxy -f[/]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
