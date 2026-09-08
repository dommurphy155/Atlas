"""Atlas proxy runtime abstraction.

Decides — at install time and at every command invocation — which mechanism
Atlas should use to run, supervise, and surface logs for the proxy process.

The runtime is selected at install time and persisted to
`<repo>/data/runtime.json` so subsequent commands (`start`, `stop`, `status`,
`logs`, `doctor`) know what to drive without re-detecting the world every
time. Re-running `atlas install` re-detects and re-selects.

Selection priority (highest first):
  1. systemd system  — Linux + root OR sudo works + systemctl + systemd PID 1
  2. systemd --user  — Linux + XDG_RUNTIME_DIR + systemctl --user + loginctl
                        session for current user
  3. tmux-equivalent — any of: tmux, psmux, tmuxw, lumux, qscn, wmux
  4. nohup          — POSIX nohup (always available on unix) — last
                       mechanical fallback. Foreground `atlas start` and
                       detached `nohup` are not equivalent: nohup survives
                       terminal close; `atlas start` foreground blocks.
  5. manual         — Windows without a tmux-equivalent. User runs the
                       proxy as a foreground process from their shell.

The "systemd system" mode is always preferred when actually usable. The
fallbacks only kick in when root / sudo / systemd genuinely are unavailable.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

OSFamily = Literal["linux", "macos", "windows", "other"]
RuntimeMode = Literal["systemd", "systemd-user", "tmux", "nohup", "manual"]

# Tmux-equivalent binaries (tmux itself + native Windows clones).
TMUX_BINARIES = ("tmux", "psmux", "tmuxw", "lumux", "qscn", "wmux")


@dataclass
class RuntimeEnv:
    """Snapshot of the host environment relevant to runtime selection.

    Frozen after detection — all fields are read-only facts, not preferences.
    """
    os: OSFamily
    is_root: bool
    sudo_works: bool
    has_systemctl: bool
    systemd_pid1: bool
    has_user_systemd: bool
    has_loginctl: bool
    tmux_bin: str | None  # first tmux-equivalent on PATH
    has_nohup: bool

    # Convenience derivations
    @property
    def systemd_system_usable(self) -> bool:
        return (
            self.os == "linux"
            and self.has_systemctl
            and self.systemd_pid1
            and (self.is_root or self.sudo_works)
        )

    @property
    def systemd_user_usable(self) -> bool:
        # Root can also use systemd --user if a user bus exists
        # (e.g. via logind). Non-root needs XDG_RUNTIME_DIR set.
        if self.os != "linux" or not self.has_systemctl:
            return False
        if not self.has_user_systemd:
            return False
        # Verify the user bus is actually reachable by checking
        # if the runtime directory exists and systemctl --user
        # can communicate with it.
        if self.is_root:
            # Root: check for a user bus (logind session).
            # XDG_RUNTIME_DIR may or may not be set for root.
            xdg = os.environ.get("XDG_RUNTIME_DIR", "")
            if not xdg:
                try:
                    xdg = f"/run/user/{os.getuid()}"
                except AttributeError:
                    return False
            return Path(xdg).is_dir() and self.has_loginctl
        # Non-root: XDG_RUNTIME_DIR must be set and valid.
        xdg = os.environ.get("XDG_RUNTIME_DIR", "")
        if not xdg:
            try:
                xdg = f"/run/user/{os.getuid()}"
            except AttributeError:
                return False
        return bool(xdg and Path(xdg).is_dir())

    @property
    def tmux_usable(self) -> bool:
        return self.tmux_bin is not None

    @property
    def nohup_usable(self) -> bool:
        return self.os in ("linux", "macos", "other") and self.has_nohup


def detect_os() -> OSFamily:
    sysname = platform.system().lower()
    if sysname == "linux":
        return "linux"
    if sysname == "darwin":
        return "macos"
    if sysname in ("windows", "win32"):
        return "windows"
    return "other"


def _run_quiet(cmd: list[str], timeout: float = 2.0) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _find_tmux() -> str | None:
    for b in TMUX_BINARIES:
        p = shutil.which(b)
        if p:
            return p
    return None


def detect_env() -> RuntimeEnv:
    """Snapshot of the current host — pure, no side effects, no I/O."""
    os_name = detect_os()
    is_root = False
    try:
        is_root = (os.geteuid() == 0)  # POSIX
    except AttributeError:
        # Windows — no geteuid. Check for Administrator via net session.
        r = _run_quiet(["net", "session"], timeout=2.0)
        is_root = bool(r and r.returncode == 0)

    # sudo -n true: succeeds (rc=0) iff sudo exists AND is allowed without
    # a password. If it would prompt for a password, we treat sudo as
    # non-functional for our purposes — better to fall back than hang.
    # Having sudo installed is enough to make systemd a viable runtime
    # candidate.  Privileged operations intentionally invoke normal sudo so
    # the user can authenticate interactively when required.
    sudo_works = bool(shutil.which("sudo"))

    has_systemctl = shutil.which("systemctl") is not None

    # systemd as PID 1 — the canonical "systemd is the system manager" check.
    systemd_pid1 = os_name == "linux" and Path("/proc/1/comm").exists() \
        and Path("/proc/1/comm").read_text().strip() == "systemd"

    # Per-user systemd: needs XDG_RUNTIME_DIR=/run/user/<uid> + systemctl --user
    # reachable. loginctl is a useful proxy for "is there a login session".
    has_user_systemd = False
    has_loginctl = shutil.which("loginctl") is not None
    if os_name == "linux" and has_systemctl and not is_root:
        xdg = os.environ.get("XDG_RUNTIME_DIR", "")
        if not xdg:
            try:
                xdg = f"/run/user/{os.getuid()}"
            except AttributeError:
                xdg = ""
        if xdg and Path(xdg).is_dir():
            r = _run_quiet(["systemctl", "--user", "is-active", "atlas-proxy.service"])
            # Don't require the service to exist; just require the call to not
            # fail with "Failed to connect to bus".
            if r and "Failed to connect" not in (r.stderr or ""):
                has_user_systemd = True

    tmux_bin = _find_tmux()
    # Windows: also check PATHEXT-less exe names; shutil.which handles that.

    # nohup is a shell builtin on some shells but also a binary on most
    # unix systems. Test for the binary form. Windows has no equivalent —
    # we'd never invoke nohup there.
    has_nohup = os_name in ("linux", "macos", "other") and shutil.which("nohup") is not None

    return RuntimeEnv(
        os=os_name,
        is_root=is_root,
        sudo_works=sudo_works,
        has_systemctl=has_systemctl,
        systemd_pid1=systemd_pid1,
        has_user_systemd=has_user_systemd,
        has_loginctl=has_loginctl,
        tmux_bin=tmux_bin,
        has_nohup=has_nohup,
    )


def choose_mode(env: RuntimeEnv) -> RuntimeMode:
    """Pick the best runtime mode for the given environment.

    Order of preference (highest first):
      1. systemd         — full system service, root or sudo
      2. systemd-user    — user-scoped systemd service
      3. tmux            — detached user tmux session
      4. nohup           — POSIX nohup, last mechanical fallback
      5. manual          — Windows without tmux — user runs it themselves
    """
    if env.systemd_system_usable:
        return "systemd"
    if env.systemd_user_usable:
        return "systemd-user"
    if env.tmux_usable:
        return "tmux"
    if env.nohup_usable:
        return "nohup"
    return "manual"


# ---------------------------------------------------------------------------
# Runtime-mode implementations
# ---------------------------------------------------------------------------
#
# A Runtime is a strategy for managing the proxy process: start it, stop it,
# check if it's running, and surface its logs. Each impl is small and
# self-contained. The CLI commands dispatch to the right one via get_runtime().


@dataclass
class RuntimeInfo:
    """One-line description of the active runtime — used by `atlas status`."""
    mode: RuntimeMode
    label: str        # "systemd" / "systemd-user" / "tmux" / "nohup" / "manual"
    session: str      # systemd unit / tmux session name / "atlas2-proxy"
    pid: int | None
    extra: dict = field(default_factory=dict)


class Runtime(Protocol):
    """Common interface for all runtime implementations."""
    info: RuntimeInfo

    def start(self) -> tuple[bool, str]: ...
    def stop(self) -> tuple[bool, str]: ...
    def status(self) -> tuple[bool, str]: ...
    def logs(self, follow: bool = True) -> int: ...
    def env_summary(self) -> dict: ...
    def cleanup(self) -> None: ...


# ---- systemd (system) -----------------------------------------------------

class SystemdRuntime:
    def __init__(self, env: RuntimeEnv, repo_root: Path, service_name: str,
                 venv_py: Path) -> None:
        self.env = env
        self.repo_root = repo_root
        self.service_name = service_name
        self.venv_py = venv_py
        self.info = RuntimeInfo(
            mode="systemd",
            label="systemd",
            session=service_name,
            pid=None,
        )

    def _sudo_prefix(self) -> list[str]:
        # Do not use --non-interactive: a normal sudo user may need to enter
        # their password during installation or service management.
        return [] if self.env.is_root else ["sudo"]

    def _run(self, *args: str) -> subprocess.CompletedProcess | None:
        cmd = self._sudo_prefix() + ["systemctl", *args]
        return _run_quiet(cmd, timeout=10.0)

    def install(self) -> tuple[bool, str]:
        unit = f"""[Unit]
Description=Atlas Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={self.repo_root}
ExecStart={self.venv_py} -m proxy.main
Restart=on-failure
RestartSec=5
EnvironmentFile=-{self.repo_root / ".env"}

[Install]
WantedBy=multi-user.target
"""
        unit_path = Path("/etc/systemd/system") / self.service_name
        tmp = self.repo_root / "data" / f".{self.service_name}.tmp"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(unit)

        try:
            r = self._run("stop", self.service_name)
            install = self._sudo_prefix() + [
                "install", "-m", "644", str(tmp), str(unit_path)
            ]
            ir = _run_quiet(install, timeout=10.0)
            if ir is None or ir.returncode != 0:
                return False, ir.stderr.strip() if ir else "failed to install systemd unit"

            rr = self._run("daemon-reload")
            if rr is None or rr.returncode != 0:
                return False, rr.stderr.strip() if rr else "systemctl daemon-reload failed"

            er = self._run("enable", self.service_name)
            if er is None or er.returncode != 0:
                return False, er.stderr.strip() if er else "systemctl enable failed"

            sr = self._run("start", self.service_name)
            if sr is None or sr.returncode != 0:
                return False, sr.stderr.strip() if sr else "systemctl start failed"

            return True, f"installed and started {self.service_name}"
        finally:
            tmp.unlink(missing_ok=True)

    def cleanup(self) -> None:
        """Best-effort: stop + disable + remove the unit. Never raises."""
        try:
            self._run("disable", "--now", self.service_name)
        except Exception:
            pass
        unit_path = Path("/etc/systemd/system") / self.service_name
        if unit_path.exists():
            install = self._sudo_prefix() + ["rm", "-f", str(unit_path)]
            _run_quiet(install, timeout=5.0)
        try:
            self._run("daemon-reload")
        except Exception:
            pass

    def start(self) -> tuple[bool, str]:
        r = self._run("start", self.service_name)
        if r is None or r.returncode != 0:
            return False, (r.stderr.strip() if r else "systemctl not available")
        return True, f"started {self.service_name}"

    def stop(self) -> tuple[bool, str]:
        r = self._run("stop", self.service_name)
        if r is None or r.returncode != 0:
            return False, (r.stderr.strip() if r else "systemctl not available")
        return True, f"stopped {self.service_name}"

    def status(self) -> tuple[bool, str]:
        r = self._run("is-active", self.service_name)
        if r is None:
            return False, "systemctl unavailable"
        active = r.stdout.strip() == "active"
        return active, r.stdout.strip()

    def logs(self, follow: bool = True) -> int:
        cmd = ["journalctl", "-u", self.service_name, "--no-pager"]
        if follow:
            cmd.append("-f")
        # journalctl usually needs sudo for system units
        cmd = self._sudo_prefix() + cmd
        try:
            return subprocess.call(cmd)
        except FileNotFoundError:
            return 1

    def env_summary(self) -> dict:
        return {"mode": "systemd", "service_name": self.service_name}


# ---- systemd --user -------------------------------------------------------

class SystemdUserRuntime:
    def __init__(self, env: RuntimeEnv, repo_root: Path, service_name: str,
                 venv_py: Path) -> None:
        self.env = env
        self.repo_root = repo_root
        self.service_name = service_name
        self.venv_py = venv_py
        self.info = RuntimeInfo(
            mode="systemd-user",
            label="systemd --user",
            session=service_name,
            pid=None,
        )

    def _run(self, *args: str) -> subprocess.CompletedProcess | None:
        cmd = ["systemctl", "--user", *args]
        return _run_quiet(cmd, timeout=10.0)

    def install(self) -> tuple[bool, str]:
        unit_dir = Path.home() / ".config" / "systemd" / "user"
        unit_path = unit_dir / self.service_name

        unit = f"""[Unit]
Description=Atlas Proxy
After=network-online.target

[Service]
Type=simple
WorkingDirectory={self.repo_root}
ExecStart={self.venv_py} -m proxy.main
Restart=on-failure
RestartSec=5
EnvironmentFile=-{self.repo_root / ".env"}

[Install]
WantedBy=default.target
"""

        unit_dir.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(unit)

        r = self._run("daemon-reload")
        if r is None or r.returncode != 0:
            return False, r.stderr.strip() if r else "systemctl --user daemon-reload failed"

        r = self._run("enable", self.service_name)
        if r is None or r.returncode != 0:
            return False, r.stderr.strip() if r else "systemctl --user enable failed"

        r = self._run("start", self.service_name)
        if r is None or r.returncode != 0:
            return False, r.stderr.strip() if r else "systemctl --user start failed"

        return True, f"installed and started {self.service_name}"

    def cleanup(self) -> None:
        """Best-effort: stop + disable + remove the user unit. Never raises."""
        try:
            self._run("disable", "--now", self.service_name)
        except Exception:
            pass
        unit_dir = Path.home() / ".config" / "systemd" / "user"
        unit_path = unit_dir / self.service_name
        if unit_path.exists():
            try:
                unit_path.unlink()
            except OSError:
                pass
        try:
            self._run("daemon-reload")
        except Exception:
            pass

    def start(self) -> tuple[bool, str]:
        r = self._run("start", self.service_name)
        if r is None or r.returncode != 0:
            return False, (r.stderr.strip() if r else "systemctl --user unavailable")
        return True, f"started {self.service_name}"

    def stop(self) -> tuple[bool, str]:
        r = self._run("stop", self.service_name)
        if r is None or r.returncode != 0:
            return False, (r.stderr.strip() if r else "systemctl --user unavailable")
        return True, f"stopped {self.service_name}"

    def status(self) -> tuple[bool, str]:
        r = self._run("is-active", self.service_name)
        if r is None:
            return False, "systemctl --user unavailable"
        active = r.stdout.strip() == "active"
        return active, r.stdout.strip()

    def logs(self, follow: bool = True) -> int:
        cmd = ["journalctl", "--user", "-u", self.service_name, "--no-pager"]
        if follow:
            cmd.append("-f")
        try:
            return subprocess.call(cmd)
        except FileNotFoundError:
            return 1

    def env_summary(self) -> dict:
        return {"mode": "systemd-user", "service_name": self.service_name}


# ---- tmux ----------------------------------------------------------------

TMUX_SESSION = "atlas"


class TmuxRuntime:
    def __init__(self, env: RuntimeEnv, repo_root: Path, tmux_bin: str,
                 venv_py: Path, log_file: Path) -> None:
        self.env = env
        self.repo_root = repo_root
        self.tmux_bin = tmux_bin
        self.venv_py = venv_py
        self.log_file = log_file
        self.session = TMUX_SESSION
        self.info = RuntimeInfo(
            mode="tmux",
            label=f"tmux ({Path(tmux_bin).name})",
            session=self.session,
            pid=None,
        )

    def _run(self, *args: str) -> subprocess.CompletedProcess | None:
        return _run_quiet([self.tmux_bin, *args], timeout=5.0)

    def start(self) -> tuple[bool, str]:
        if self.status()[0]:
            return True, f"already running in tmux session '{self.session}'"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # -d: detach. -s: session name. The proxy is launched inside the
        # tmux session so its stdout/stderr go to the tmux scrollback; we
        # ALSO tee to a log file for `atlas logs` to tail.
        cmd = (
            f"exec {self.venv_py} -m proxy.main 2>&1 | "
            f"tee -a {shlex_quote(str(self.log_file))}"
        )
        r = self._run(
            "new-session", "-d", "-s", self.session, "-c", str(self.repo_root),
            "bash", "-lc", cmd,
        )
        if r is None or r.returncode != 0:
            return False, (r.stderr.strip() if r else f"{self.tmux_bin} failed")
        return True, f"started in tmux session '{self.session}'"

    def stop(self) -> tuple[bool, str]:
        if not self.status()[0]:
            return True, "not running"
        r = self._run("kill-session", "-t", self.session)
        if r is None or r.returncode != 0:
            return False, (r.stderr.strip() if r else f"{self.tmux_bin} kill failed")
        return True, f"killed tmux session '{self.session}'"

    def cleanup(self) -> None:
        """Best-effort: kill the tmux session. Never raises."""
        try:
            self._run("kill-session", "-t", self.session)
        except Exception:
            pass

    def status(self) -> tuple[bool, str]:
        r = self._run("has-session", "-t", self.session)
        if r is None:
            return False, f"{self.tmux_bin} unavailable"
        return r.returncode == 0, \
            (f"running in tmux '{self.session}'" if r.returncode == 0 else "stopped")

    def logs(self, follow: bool = True) -> int:
        # Tail the tee'd log file. tmux itself has capture-pane but a flat
        # file is more portable across tmux equivalents.
        if not self.log_file.exists():
            print(f"No log file yet: {self.log_file}", file=sys.stderr)
            return 1
        cmd = ["tail", "-n", "40"]
        if follow:
            cmd.append("-f")
        cmd.append(str(self.log_file))
        try:
            return subprocess.call(cmd)
        except FileNotFoundError:
            return 1

    def env_summary(self) -> dict:
        return {
            "mode": "tmux",
            "tmux_bin": self.tmux_bin,
            "session": self.session,
            "log_file": str(self.log_file),
        }


# ---- nohup (POSIX fallback) ----------------------------------------------

class NohupRuntime:
    """Last mechanical fallback: nohup + pidfile. Foreground mode for manual."""

    def __init__(self, env: RuntimeEnv, repo_root: Path, venv_py: Path,
                 pid_file: Path, log_file: Path) -> None:
        self.env = env
        self.repo_root = repo_root
        self.venv_py = venv_py
        self.pid_file = pid_file
        self.log_file = log_file
        self.info = RuntimeInfo(
            mode="nohup",
            label="nohup (user process)",
            session="atlas-proxy",
            pid=None,
        )

    def start(self) -> tuple[bool, str]:
        if self.status()[0]:
            return True, f"already running (pid={self._pid()})"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("ab") as fh:
            proc = subprocess.Popen(
                [str(self.venv_py), "-m", "proxy.main"],
                stdout=fh, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(self.repo_root),
                start_new_session=True,
            )
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(proc.pid))
        return True, f"started (pid={proc.pid}, log={self.log_file})"

    def stop(self) -> tuple[bool, str]:
        pid = self._pid()
        if pid is None:
            self.pid_file.unlink(missing_ok=True)
            return True, "not running"
        try:
            os.kill(pid, 15)  # SIGTERM
        except ProcessLookupError:
            self.pid_file.unlink(missing_ok=True)
            return True, "not running"
        # Wait briefly for graceful exit
        for _ in range(20):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                break
        else:
            try:
                os.kill(pid, 9)  # SIGKILL fallback
            except ProcessLookupError:
                pass
        self.pid_file.unlink(missing_ok=True)
        return True, f"stopped (pid={pid})"

    def status(self) -> tuple[bool, str]:
        pid = self._pid()
        if pid is None:
            return False, "not running"
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            self.pid_file.unlink(missing_ok=True)
            return False, "not running"
        return True, f"running (pid={pid})"

    def logs(self, follow: bool = True) -> int:
        if not self.log_file.exists():
            print(f"No log file yet: {self.log_file}", file=sys.stderr)
            return 1
        cmd = ["tail", "-n", "40"]
        if follow:
            cmd.append("-f")
        cmd.append(str(self.log_file))
        try:
            return subprocess.call(cmd)
        except FileNotFoundError:
            return 1

    def _pid(self) -> int | None:
        if not self.pid_file.exists():
            return None
        try:
            return int(self.pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    def cleanup(self) -> None:
        """Best-effort: stop + remove pidfile. Never raises."""
        try:
            self.stop()
        except Exception:
            pass
        self.pid_file.unlink(missing_ok=True)

    def env_summary(self) -> dict:
        return {
            "mode": "nohup",
            "pid_file": str(self.pid_file),
            "log_file": str(self.log_file),
        }


# ---- manual (Windows / last resort) --------------------------------------

class ManualRuntime:
    """No persistent supervisor available — user runs `atlas start` fg."""

    def __init__(self) -> None:
        self.info = RuntimeInfo(
            mode="manual",
            label="manual (foreground)",
            session="atlas-proxy",
            pid=None,
        )

    def start(self) -> tuple[bool, str]:
        return False, (
            "No persistent supervisor available on this system.\n"
            "Run the proxy manually:  cd <repo> && .venv/bin/python -m proxy.main"
        )

    def stop(self) -> tuple[bool, str]:
        return False, "No persistent supervisor — manually Ctrl-C the running process."

    def status(self) -> tuple[bool, str]:
        return False, "manual mode — no supervisor tracks the process"

    def logs(self, follow: bool = True) -> int:
        print("No supervisor to read logs from. Start the proxy manually to see its output.", file=sys.stderr)
        return 1

    def cleanup(self) -> None:
        """Nothing to clean up — manual mode owns no process."""
        return None

    def env_summary(self) -> dict:
        return {"mode": "manual"}


# ---------------------------------------------------------------------------
# Persistence: data/runtime.json
# ---------------------------------------------------------------------------

RUNTIME_FILE_NAME = "runtime.json"


def _runtime_path(repo_root: Path) -> Path:
    return repo_root / "data" / RUNTIME_FILE_NAME


def load_runtime_choice(repo_root: Path) -> dict | None:
    """Read the previously chosen runtime — None if not yet installed."""
    p = _runtime_path(repo_root)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_runtime_choice(repo_root: Path, env: RuntimeEnv, mode: RuntimeMode,
                        info: dict) -> None:
    """Persist the install-time choice for subsequent commands."""
    p = _runtime_path(repo_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "env": asdict(env),
        "info": info,
        "installed_at": int(time.time()),
    }
    p.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Factory: build the right Runtime for the current situation
# ---------------------------------------------------------------------------

def _runtime_paths(repo_root: Path, venv_py: Path):
    """Compute per-runtime paths (pidfile, log file) — kept centralised."""
    data_dir = repo_root / "data"
    return {
        "pid_file": data_dir / "run.pid",
        "log_file": data_dir / "run.log",
        "venv_py": venv_py,
        "repo_root": repo_root,
    }


def install_runtime(repo_root: Path, venv_py: Path,
                   service_name: str) -> tuple[RuntimeMode, bool, str]:
    """Detect, attempt, verify, fallback. Only persist a runtime after a
    successful health check.

    The flow is:
      1. Detect the environment.
      2. Build an ordered list of runtime candidates.
      3. For each candidate: install (or start), then poll the proxy's
         health endpoint with a short bounded budget.
      4. If health succeeds: persist the choice and return.
      5. If health fails: cleanup the candidate and try the next.
      6. If every candidate fails: persist ``manual`` and return success
         so the installer itself never fails on a runtime problem.
    """
    env = detect_env()
    candidates = _candidate_modes(env)

    def _emit(msg: str) -> None:
        # Progress lines are consumed by the shell installer (which indents
        # them) and shown raw when invoked from the Python wizard.
        print(msg, flush=True)

    _emit(f"Detecting runtime — candidates: {', '.join(candidates) or 'none'}")

    last_message = "no runtime candidates available"
    first = True
    for mode in candidates:
        if first:
            _emit(f"Runtime: {mode}")
            first = False
        else:
            _emit(f"Falling back: {mode}")
        runtime = build_runtime(mode, env, repo_root, venv_py, service_name)
        installer = getattr(runtime, "install", None)
        try:
            ok, message = installer() if callable(installer) else runtime.start()
        except Exception as exc:  # installer crashed — treat as failure
            ok, message = False, f"{type(exc).__name__}: {exc}"

        if not ok:
            _emit(f"  ! {mode} failed: {message}")
            # Best-effort cleanup before falling back. Never raises.
            try:
                runtime.cleanup()
            except Exception:
                pass
            last_message = f"{mode}: {message}"
            continue

        # Install/start succeeded — now verify the proxy is actually healthy.
        healthy, health_msg = check_health(repo_root=repo_root)
        if not healthy:
            _emit(f"  ! {mode} unhealthy: {health_msg}")
            try:
                runtime.cleanup()
            except Exception:
                pass
            last_message = f"{mode}: {health_msg}"
            continue

        _emit(f"  ✓ {mode} healthy: {health_msg}")
        save_runtime_choice(
            repo_root, env, mode, runtime.info.__dict__,
        )
        return mode, True, health_msg or message

    # Everything failed — record manual so the installer still completes.
    _emit("! No automatic runtime manager available")
    manual_rt = build_runtime("manual", env, repo_root, venv_py, service_name)
    save_runtime_choice(
        repo_root, env, "manual", manual_rt.info.__dict__,
    )
    return "manual", True, (
        f"no runtime worked ({last_message}); persisted manual mode"
    )


# Health endpoint + bounded polling window. A few seconds is plenty for
# uvicorn to bind; we don't want the installer sitting around for minutes.
#
# NOTE: the fork's default LISTEN_PORT is 8777 (prod lives on 8788).
# We prefer reading it from proxy.config so we always check the right port;
# a hard-coded constant is only used as a last-resort fallback.
HEALTH_PORT_DEFAULT = 8777
HEALTH_TIMEOUT_S = 6.0
HEALTH_POLL_S = 0.25


def _health_url(repo_root: Path) -> str:
    """Resolve the actual proxy port for this repo.

    Priority:
      1. ``data/bound_port`` — written by the running proxy with the port
         it actually bound to. Honors port-collision fallback at runtime.
      2. ``proxy.config.LISTEN_PORT`` — the configured default (fork: 8777,
         NOT prod's 8788).
      3. The fork's compiled-in default.
    """
    port: int | None = None
    bp = repo_root / "data" / "bound_port"
    if bp.exists():
        try:
            v = int(bp.read_text().strip())
            if 1 <= v <= 65535:
                port = v
        except (ValueError, OSError):
            pass
    if port is None:
        try:
            sys.path.insert(0, str(repo_root))
            from proxy.config import LISTEN_PORT  # type: ignore
            port = int(LISTEN_PORT)
        except Exception:
            pass
    if port is None:
        port = HEALTH_PORT_DEFAULT
    return f"http://127.0.0.1:{port}/health"


def check_health(timeout_s: float = HEALTH_TIMEOUT_S,
                 url: str | None = None,
                 repo_root: Path | None = None) -> tuple[bool, str]:
    """Poll the proxy health endpoint with a short bounded budget.

    Returns (True, "ok") on success or (False, reason) on timeout/error.
    Uses urllib so we don't pull requests into the installer path.
    """
    import socket
    import urllib.request
    import urllib.error

    if url is None:
        if repo_root is None:
            url = f"http://127.0.0.1:{HEALTH_PORT_DEFAULT}/health"
        else:
            url = _health_url(repo_root)

    deadline = time.monotonic() + timeout_s
    last_err = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                if 200 <= resp.status < 300:
                    return True, f"health check passed ({url})"
                last_err = f"status {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = getattr(exc, "reason", None) or str(exc) or "unreachable"
        except socket.timeout:
            last_err = "timeout"
        time.sleep(HEALTH_POLL_S)
    return False, f"health check failed: {last_err or 'no response'} ({url})"


def _candidate_modes(env: RuntimeEnv) -> list[RuntimeMode]:
    """Build the ordered list of runtime candidates for this environment.

    Linux:  systemd-system -> systemd-user -> tmux -> nohup
    macOS:  tmux -> nohup
    Other:  tmux -> nohup
    Manual is always the final fallback (handled by the caller).
    """
    if env.os == "linux":
        candidates: list[RuntimeMode] = []
        if env.systemd_system_usable:
            candidates.append("systemd")
        if env.systemd_user_usable:
            candidates.append("systemd-user")
        if env.tmux_usable:
            candidates.append("tmux")
        if env.nohup_usable:
            candidates.append("nohup")
        return candidates
    candidates = []
    if env.tmux_usable:
        candidates.append("tmux")
    if env.nohup_usable:
        candidates.append("nohup")
    return candidates


def build_runtime(mode: RuntimeMode, env: RuntimeEnv, repo_root: Path,
                  venv_py: Path, service_name: str) -> Runtime:
    """Construct the Runtime implementation for the chosen mode."""
    paths = _runtime_paths(repo_root, venv_py)
    if mode == "systemd":
        return SystemdRuntime(env, repo_root, service_name, paths["venv_py"])
    if mode == "systemd-user":
        return SystemdUserRuntime(env, repo_root, service_name, paths["venv_py"])
    if mode == "tmux":
        tmux_bin = env.tmux_bin or shutil.which("tmux") or "tmux"
        return TmuxRuntime(env, repo_root, tmux_bin, paths["venv_py"],
                           paths["log_file"])
    if mode == "nohup":
        return NohupRuntime(env, repo_root, paths["venv_py"],
                            paths["pid_file"], paths["log_file"])
    if mode == "manual":
        return ManualRuntime()
    raise ValueError(f"unknown runtime mode: {mode!r}")


def get_runtime(repo_root: Path, venv_py: Path, service_name: str) -> Runtime:
    """Pick the active Runtime for subsequent commands.

    Strategy:
      1. If `data/runtime.json` exists AND the persisted mode is still
         actually usable on this host, honour it.
      2. Otherwise re-detect from the environment and pick the best
         available mode (with ``manual`` as the last-resort fallback).

    This is intentionally forgiving: a host that previously had
    systemd-user but lost its user-bus between installs will not get
    stuck trying to use a runtime that no longer exists.
    """
    persisted = load_runtime_choice(repo_root)
    env = detect_env()

    if persisted:
        mode = persisted.get("mode", "nohup")
        if _mode_usable(mode, env):
            return build_runtime(mode, env, repo_root, venv_py, service_name)

    # Re-detect from the live environment.
    candidates = _candidate_modes(env)
    for mode in candidates:
        if _mode_usable(mode, env):
            return build_runtime(mode, env, repo_root, venv_py, service_name)

    return build_runtime("manual", env, repo_root, venv_py, service_name)


def _mode_usable(mode: RuntimeMode, env: RuntimeEnv) -> bool:
    """Is this mode actually usable on the current host right now?"""
    if mode == "systemd":
        return env.systemd_system_usable
    if mode == "systemd-user":
        return env.systemd_user_usable
    if mode == "tmux":
        return env.tmux_usable
    if mode == "nohup":
        return env.nohup_usable
    if mode == "manual":
        return True
    return False


# ---------------------------------------------------------------------------
# Tiny helpers used by the wizard
# ---------------------------------------------------------------------------

def shlex_quote(s: str) -> str:
    """Quote a string for safe inclusion in a shell command."""
    import shlex
    return shlex.quote(s)
