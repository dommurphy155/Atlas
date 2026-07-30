"""File system operations with safety checks."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .console import get_console
from .platform import PlatformInfo, detect_platform, get_sudo_prefix


console = get_console()


def ensure_dir(path: Path, mode: int = 0o755) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    return path


def write_file(path: Path, content: str, mode: int = 0o644, backup: bool = True) -> bool:
    """Write file atomically with optional backup."""
    if path.exists():
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
            console.debug(f"Backed up {path} -> {backup_path}")
        else:
            console.warning(f"Overwriting {path}")
    else:
        ensure_dir(path.parent)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.chmod(mode)
        tmp_path.replace(path)
        return True
    except Exception as e:
        console.error(f"Failed to write {path}: {e}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False


def read_file(path: Path) -> Optional[str]:
    """Read file safely."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def copy_template(src: Path, dst: Path, substitutions: dict[str, str]) -> bool:
    """Copy template file with placeholder substitution."""
    content = read_file(src)
    if content is None:
        console.error(f"Template not found: {src}")
        return False

    for key, value in substitutions.items():
        content = content.replace(f"__{key}__", value)

    return write_file(dst, content)


def run_command(
    cmd: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    capture: bool = False,
    check: bool = True,
    timeout: Optional[int] = None,
    sudo: bool = False,
) -> subprocess.CompletedProcess:
    """Run command with consistent handling."""
    if sudo:
        cmd = get_sudo_prefix() + cmd

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    console.debug(f"$ {' '.join(cmd)}" + (f" (cwd={cwd})" if cwd else ""))

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            capture_output=capture,
            text=capture,
            check=check,
            timeout=timeout,
        )
        if capture and result.stdout:
            console.debug(result.stdout.strip())
        return result
    except subprocess.CalledProcessError as e:
        console.error(f"Command failed: {' '.join(cmd)}")
        if e.stdout:
            console.debug(f"stdout: {e.stdout.strip()}")
        if e.stderr:
            console.error(f"stderr: {e.stderr.strip()}")
        raise
    except subprocess.TimeoutExpired:
        console.error(f"Command timed out: {' '.join(cmd)}")
        raise


def run_shell(
    script: str,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run shell script (bash on Unix, cmd on Windows) — DEPRECATED: use run_command with explicit args."""
    import warnings
    warnings.warn("run_shell is deprecated — use run_command with explicit argv list", DeprecationWarning, stacklevel=2)
    platform = detect_platform()
    if platform.is_windows:
        return run_command(["cmd", "/c", script], cwd=cwd, env=env, check=check, shell=False)
    return run_command(["bash", "-c", script], cwd=cwd, env=env, check=check, shell=False)


def symlink(target: Path, link: Path, sudo: bool = False) -> bool:
    """Create symlink with platform handling."""
    try:
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == target.resolve():
                console.debug(f"Symlink already correct: {link} -> {target}")
                return True
            if sudo:
                run_command(["rm", "-f", str(link)], sudo=True)
            else:
                link.unlink(missing_ok=True)

        ensure_dir(link.parent)
        if sudo:
            run_command(["ln", "-sf", str(target), str(link)], sudo=True)
        else:
            link.symlink_to(target)
        console.success(f"Linked {link} -> {target}")
        return True
    except Exception as e:
        console.error(f"Failed to create symlink {link} -> {target}: {e}")
        return False


def which(cmd: str) -> Optional[Path]:
    """Find executable in PATH."""
    from .platform import find_executable
    return find_executable(cmd)


def append_to_file(path: Path, lines: list[str], marker: str = "# --- Atlas ---") -> bool:
    """Append lines to file if marker not present."""
    content = read_file(path) or ""
    if marker in content:
        console.debug(f"Marker already in {path}, skipping")
        return True

    new_content = content.rstrip() + "\n\n" + marker + "\n" + "\n".join(lines) + "\n# --- end Atlas ---\n"
    return write_file(path, new_content)


def remove_from_file(path: Path, marker: str = "# --- Atlas ---") -> bool:
    """Remove block between markers."""
    content = read_file(path)
    if content is None or marker not in content:
        return True

    lines = content.splitlines()
    output = []
    in_block = False
    for line in lines:
        if marker in line and "end Atlas" not in line:
            in_block = True
            continue
        if "end Atlas" in line:
            in_block = False
            continue
        if not in_block:
            output.append(line)

    return write_file(path, "\n".join(output))