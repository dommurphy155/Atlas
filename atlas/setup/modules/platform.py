"""Platform detection and paths."""

import os
import platform as plat
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PlatformInfo:
    """Detected platform information."""
    system: str  # "linux", "darwin", "windows"
    shell: str   # "bash", "zsh", "fish", "cmd", "powershell"
    is_linux: bool = False
    is_macos: bool = False
    is_windows: bool = False
    is_unix: bool = False

    # Paths
    home: Path = field(default_factory=Path.home)
    project_dir: Path = field(default_factory=Path.cwd)
    venv_dir: Path = field(default_factory=lambda: Path.cwd() / ".venv")
    venv_bin: Path = field(default_factory=lambda: Path.cwd() / ".venv" / "bin")
    bin_dir: Path = field(default_factory=lambda: Path("/usr/local/bin"))

    # User
    default_user: str = "root"


def detect_platform() -> PlatformInfo:
    """Detect current platform and return PlatformInfo."""
    system = plat.system().lower()

    if system == "linux":
        is_linux, is_macos, is_windows = True, False, False
        is_unix = True
        bin_dir = Path("/usr/local/bin")
        default_user = _detect_linux_user()
        venv_bin = Path(".venv") / "bin"
    elif system == "darwin":
        is_linux, is_macos, is_windows = False, True, False
        is_unix = True
        bin_dir = Path("/usr/local/bin")
        default_user = os.getenv("SUDO_USER") or os.getenv("USER") or "root"
        venv_bin = Path(".venv") / "bin"
    elif system == "windows":
        is_linux, is_macos, is_windows = False, False, True
        is_unix = False
        bin_dir = Path(os.getenv("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        default_user = os.getenv("USERNAME", "Administrator")
        venv_bin = Path(".venv") / "Scripts"
    else:
        is_linux, is_macos, is_windows = False, False, False
        is_unix = True
        bin_dir = Path("/usr/local/bin")
        default_user = "root"
        venv_bin = Path(".venv") / "bin"

    # Detect shell
    shell = _detect_shell()

    # Resolve paths
    project_dir = Path.cwd()
    venv_dir = project_dir / ".venv"
    venv_bin = venv_dir / ("Scripts" if is_windows else "bin")

    return PlatformInfo(
        system=system,
        shell=shell,
        is_linux=is_linux,
        is_macos=is_macos,
        is_windows=is_windows,
        is_unix=is_unix,
        home=Path.home(),
        project_dir=project_dir,
        venv_dir=venv_dir,
        venv_bin=venv_bin,
        bin_dir=bin_dir,
        default_user=default_user,
    )


def _detect_linux_user() -> str:
    """Detect the actual user on Linux (handles sudo)."""
    return os.getenv("SUDO_USER") or os.getenv("USER") or "root"


def _detect_shell() -> str:
    """Detect user's shell."""
    shell_path = os.getenv("SHELL", "")
    shell = Path(shell_path).name if shell_path else ""

    if shell in ("bash", "zsh", "fish", "sh"):
        return shell

    # Check what's available
    if shutil.which("zsh"):
        return "zsh"
    if shutil.which("fish"):
        return "fish"
    return "bash"


def find_executable(name: str) -> Optional[Path]:
    """Find executable in PATH."""
    path = shutil.which(name)
    return Path(path) if path else None


def get_sudo_prefix() -> list[str]:
    """Get sudo prefix if not root."""
    if is_root():
        return []
    return ["sudo"]


def is_root() -> bool:
    """Check if running as root."""
    return os.geteuid() == 0 if hasattr(os, 'geteuid') else False