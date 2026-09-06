"""Atlas proxy + coding-harness setup wizard.

Replaces the previous bash-only installer with an interactive Python flow:

  1. Bootstrap proxy (venv, deps, dirs, runtime, CLI symlink)
  2. Detect OS
  3. Pick a coding harness
  4. Silently detect/install the harness (background-style spinner in UI)
  5. In parallel (visually), pick API provider + import keys if needed
  6. Configure the harness to use the Atlas proxy (with backups, min changes)
  7. Smoke-test proxy + harness (real one-shot where supported; --version fallback)
  8. Launch the harness (interactive TTY) or print the launch command

Designed so the user sees a clean UI — all install noise, dependency
downloads, config file writes happen behind a spinner.

Harness install + config research is verified against current official docs
(see comments next to each `_install_*` / `_configure_*` function).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

CONSOLE = Console()
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # atlas/bin/setup_wizard -> repo root
PROXY_PKG = REPO_ROOT / "proxy"
DATA_DIR = REPO_ROOT / "data"
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
SERVICE_NAME = os.environ.get("ATLAS_SERVICE_NAME", "atlas-proxy.service")
INSTALLER = REPO_ROOT / "setup" / "install.sh"
CLI_BIN_SRC = REPO_ROOT / "atlas" / "bin" / "atlas"

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def _g(t):  return f"[green]{t}[/green]"
def _c(t):  return f"[cyan]{t}[/cyan]"
def _y(t):  return f"[yellow]{t}[/yellow]"
def _r(t):  return f"[red]{t}[/red]"
def _dim(t): return f"[dim]{t}[/dim]"
def _check(ok, t): return _g("✓") + f" {t}" if ok else _r("✗") + f" {t}"


# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------
def detect_os() -> str:
    sysname = platform.system().lower()
    if sysname == "linux":
        return "linux"
    if sysname == "darwin":
        return "macos"
    if sysname == "windows":
        return "windows"
    return sysname


# ---------------------------------------------------------------------------
# Bootstrap (proxy install: dirs, venv, deps, systemd, CLI symlink)
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    for d in ("openrouter_data", "huggingface_data", "proxy_data", "payloads"):
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)
    (PROXY_PKG / "logs").mkdir(parents=True, exist_ok=True)


def ensure_venv_and_deps(spinner_live) -> bool:
    if not VENV_PY.exists():
        spinner_live.update(_c("[1/4] creating venv..."))
        subprocess.run([sys.executable, "-m", "venv", str(REPO_ROOT / ".venv")], check=True)
        spinner_live.update(_c("[2/4] upgrading pip..."))
        subprocess.run([str(VENV_PY), "-m", "pip", "install", "-q", "--upgrade", "pip", "wheel"], check=True)
    else:
        spinner_live.update(_c("[1/4] venv present, checking deps..."))
    spinner_live.update(_c("[2/4] installing requirements..."))
    req = REPO_ROOT / "requirements.txt"
    if req.exists():
        subprocess.run([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(req)], check=True)
    spinner_live.update(_c("[3/4] verifying imports..."))
    r = subprocess.run(
        [str(VENV_PY), "-c", "import fastapi, uvicorn, httpx, orjson"],
        check=False, capture_output=True, text=True,
    )
    if r.returncode != 0:
        CONSOLE.print(_r("deps missing after install"))
        if r.stderr:
            CONSOLE.print(_dim(r.stderr.strip()))
        return False
    return True


def ensure_cli_symlink(spinner_live) -> None:
    """Install the CLI into a directory on the user's PATH.

    The CLI binary is named `atlas` (display name) but is invoked as
    `atlas` on this machine (per the project's local-command convention).
    Prefers system scope (`/usr/local/bin/atlas`) when root/sudo is
    available AND the directory exists. Otherwise falls back to
    `~/.local/bin/atlas` (XDG-conformant user-scoped location) and emits
    an opt-in hint about adding it to PATH.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "atlas"))
    from bin import runtime as _rt
    env = _rt.detect_env()
    sys_dest = Path("/usr/local/bin/atlas")
    user_dest = Path(os.path.expanduser("~/.local/bin/atlas"))

    # System scope if we can write there
    if (env.is_root or env.sudo_works) and _dir_writable(sys_dest.parent):
        dest = sys_dest
    else:
        dest = user_dest

    if dest.exists() and not dest.is_symlink():
        spinner_live.update(_y(f"CLI bin: {dest} exists, not ours — leaving alone"))
        return
    if dest.is_symlink():
        try:
            current = os.readlink(dest)
            if Path(current).resolve() == CLI_BIN_SRC.resolve():
                spinner_live.update(_g(f"CLI bin: {dest.name} already correct"))
                return
        except OSError:
            pass
        dest.unlink()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(CLI_BIN_SRC)
        spinner_live.update(_g(f"CLI bin: {dest} → atlas/bin/atlas"))
        if dest == user_dest:
            bashrc = Path(os.path.expanduser("~/.bashrc"))
            if bashrc.exists():
                content = bashrc.read_text()
                marker = "# Atlas proxy installer: CLI on PATH"
                if marker not in content:
                    bashrc.write_text(
                        content + f"\n{marker}\nexport PATH=$HOME/.local/bin:$PATH\n"
                    )
                    spinner_live.update(_y(f"  added ~/.local/bin to PATH in {bashrc}"))
    except PermissionError as e:
        spinner_live.update(_y(f"CLI bin: couldn't write {dest}: {e}"))


def _dir_writable(path: Path) -> bool:
    if not path.exists():
        return False
    return os.access(path, os.W_OK)


def ensure_runtime(spinner_live, dry_run: bool) -> str | None:
    """Select and install the best available Atlas runtime."""
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "atlas"))
    from bin import runtime as _rt

    env = _rt.detect_env()
    mode = _rt.choose_mode(env)

    spinner_live.update(_c(f"[4/4] runtime: {mode}"))

    if dry_run:
        return mode

    try:
        selected, ok, message = _rt.install_runtime(
            REPO_ROOT,
            VENV_PY,
            SERVICE_NAME,
        )

        if not ok:
            spinner_live.update(
                _r(f"[4/4] runtime failed: {message}")
            )
            return None

        spinner_live.update(
            _g(f"[4/4] runtime installed: {selected}")
        )
        return selected

    except Exception as exc:
        spinner_live.update(_r(f"[4/4] runtime failed: {exc}"))
        return None

def ensure_dotenv(spinner_live) -> None:
    env = REPO_ROOT / ".env"
    example = REPO_ROOT / ".env.example"
    if env.exists():
        spinner_live.update(_g(".env present"))
        return
    if example.exists():
        env.write_text(example.read_text())
        spinner_live.update(_y(".env created from .env.example — edit keys when ready"))


def bootstrap_proxy(dry_run: bool) -> bool:
    """Bootstrap dirs + venv + deps + runtime (systemd/tmux/nohup) + CLI symlink.

    Runtime detection happens inside `ensure_runtime` — no caller-side
    branch on sudo/systemd/tmux.
    """
    if dry_run:
        env = _detect_env_inline()
        CONSOLE.print(_dim(f"[bootstrap] dry-run — OS={env.os} root={env.is_root} sudo={env.sudo_works}"))
        # Still run ensure_runtime in dry-run so we surface the chosen mode
        with Live(Spinner("dots", text=Text("")), console=CONSOLE, transient=True) as live:
            ensure_runtime(live, dry_run=True)
        return True
    ensure_dirs()
    with Live(Spinner("dots", text=Text("")), console=CONSOLE, transient=True) as live:
        ok = ensure_venv_and_deps(live)
        if not ok:
            live.update(_r("deps missing"))
            return False
        ensure_runtime(live, dry_run=False)
        ensure_cli_symlink(live)
        live.update(_g("bootstrap complete"))
    return True


def _detect_env_inline():
    """Thin local helper — import the runtime module to keep bootstrap isolated."""
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "atlas"))
    from bin import runtime as _rt
    return _rt.detect_env()


# ===========================================================================
# HARNESS INSTALL — verified against current official docs (2026)
# ===========================================================================

def _start_claude_code_install():
    """Start Claude Code installation in the background.

    Returns the subprocess handle and installation environment.
    The caller must wait for completion with _finish_claude_code_install().
    """
    install_dir = Path.home() / ".local" / "bin"
    install_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("curl") is None:
        CONSOLE.print("  [red]✗ curl is required to install Claude Code[/red]")
        return None

    env = os.environ.copy()
    env["PATH"] = f"{install_dir}:{env.get('PATH', '')}"

    try:
        process = subprocess.Popen(
            [
                "bash",
                "-c",
                "curl -fsSL https://claude.ai/install.sh | bash",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except OSError as exc:
        CONSOLE.print(
            f"  [red]✗ Failed to start Claude Code installer: {exc}[/red]"
        )
        return None

    return process, env, install_dir


def _finish_claude_code_install(state) -> bool:
    """Wait for background Claude installation and fully verify it."""
    if state is None:
        return False

    process, env, install_dir = state

    CONSOLE.print("  Checking Claude Code installation...")

    try:
        _, stderr = process.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        process.kill()
        _, stderr = process.communicate()

        CONSOLE.print(
            "  [red]✗ Claude Code installation timed out[/red]"
        )
        return False

    if process.returncode != 0:
        CONSOLE.print(
            f"  [red]✗ Claude Code installer failed "
            f"(exit {process.returncode})[/red]"
        )

        for line in (stderr or "").strip().splitlines()[-8:]:
            CONSOLE.print(f"  [red]{line}[/red]")

        return False

    claude_path = install_dir / "claude"

    if not claude_path.is_file():
        resolved = shutil.which("claude", path=env["PATH"])
        if resolved:
            claude_path = Path(resolved)

    if not claude_path.is_file():
        CONSOLE.print(
            "  [red]✗ Claude Code installer finished but "
            "the executable was not found[/red]"
        )
        return False

    try:
        verify = subprocess.run(
            [str(claude_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        CONSOLE.print(
            f"  [red]✗ Claude Code verification failed: {exc}[/red]"
        )
        return False

    if verify.returncode != 0:
        CONSOLE.print(
            f"  [red]✗ Claude Code failed verification "
            f"(exit {verify.returncode})[/red]"
        )

        for line in (verify.stderr or "").strip().splitlines()[-5:]:
            CONSOLE.print(f"  [red]{line}[/red]")

        return False

    os.environ["PATH"] = env["PATH"]

    version = (verify.stdout or "").strip()
    version = version.splitlines()[-1] if version else "version unknown"

    CONSOLE.print(
        f"  [green]✓ Claude Code verified: {version}[/green]"
    )
    CONSOLE.print(
        f"  [dim]Executable: {claude_path}[/dim]"
    )

    return True


def _install_claude_code() -> bool:
    """Install Claude Code and wait until it is fully verified."""
    state = _start_claude_code_install()

    if state is None:
        return False

    return _finish_claude_code_install(state)


def _install_codex() -> bool:
    """Codex CLI — official installer (preferred) + npm + brew fallbacks.

    Source: https://github.com/openai/codex (README).
    Priority order (all official):
      1. curl -fsSL https://chatgpt.com/codex/install.sh | sh   (Mac/Linux)
      2. npm install -g @openai/codex                              (works, npm
            postinstall downloads the Rust binary)
      3. brew install --cask codex                                (Mac)
    We try them in order; first success wins.
    """
    # Already installed
    if shutil.which("codex"):
        return True

    # 1. Official one-liner
    if shutil.which("curl") and detect_os() in ("linux", "macos"):
        r = subprocess.run(
            ["bash", "-c", "curl -fsSL https://chatgpt.com/codex/install.sh | sh"],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0 and shutil.which("codex"):
            return True

    # 2. npm fallback (still works — downloads Rust binary on postinstall)
    if shutil.which("npm"):
        r = subprocess.run(
            ["npm", "install", "-g", "@openai/codex"],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0 and shutil.which("codex"):
            return True

    # 3. Homebrew fallback (Mac only)
    if detect_os() == "macos" and shutil.which("brew"):
        subprocess.run(
            ["brew", "install", "--cask", "codex"],
            capture_output=True, text=True, timeout=300,
        )
        if shutil.which("codex"):
            return True

    return False


def _install_hermes() -> bool:
    """Hermes Agent — official installer.

    Source: https://hermes-agent.nousresearch.com/docs/getting-started/installation
    Linux/macOS/WSL2: curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
    Windows:          iex (irm https://hermes-agent.nousresearch.com/install.ps1)

    --non-interactive avoids the interactive setup wizard at the end so the
    proxy-wiring we do next isn't undone.
    """
    os_name = detect_os()
    if os_name == "windows":
        # PowerShell one-liner. Requires PowerShell on PATH as 'pwsh' or 'powershell'.
        ps = shutil.which("pwsh") or shutil.which("powershell") or shutil.which("powershell.exe")
        if not ps:
            return False
        cmd = (
            f"{ps} -ExecutionPolicy ByPass -c "
            "\"iex (irm https://hermes-agent.nousresearch.com/install.ps1)\""
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        return r.returncode == 0 and shutil.which("hermes") is not None

    # Linux / macOS / WSL2 / Termux
    if shutil.which("curl") is None:
        return False
    cmd = (
        "curl -fsSL https://hermes-agent.nousresearch.com/install.sh "
        "| bash -s -- --non-interactive"
    )
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=600)
    return r.returncode == 0 and shutil.which("hermes") is not None


def _install_pi() -> bool:
    """Pi — official npm package.

    Source: https://www.npmjs.com/package/@earendil-works/pi-coding-agent
    """
    if shutil.which("npm") is None:
        return False
    r = subprocess.run(
        ["npm", "install", "-g", "@earendil-works/pi-coding-agent"],
        capture_output=True, text=True, timeout=300,
    )
    return r.returncode == 0 and shutil.which("pi") is not None


# ===========================================================================
# HARNESS CONFIGURE — verified against each project's real config mechanism
# ===========================================================================

def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _backup(path: Path) -> Path | None:
    """Timestamped backup. NEVER overwrites; returns the backup path."""
    if not path.exists():
        return None
    ts = _timestamp()
    backup = path.with_name(f"{path.name}.atlas-bak.{ts}")
    # If two backups land in the same second, suffix them
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.atlas-bak.{ts}.{suffix}")
        suffix += 1
    shutil.copy2(path, backup)
    return backup


def _deep_merge(target: dict, patch: dict) -> dict:
    """Recursively merge patch into target; lists/scalars replaced, dicts merged."""
    out = dict(target)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


# ---- Claude Code ----------------------------------------------------------
def _configure_claude(base_url: str, dummy_key: str) -> tuple[bool, str]:
    """Claude Code — write to the `env` block of ~/.claude/settings.json.

    Source: https://code.claude.com/docs/en/llm-gateway-connect
    "Set in a settings file ... the `env` block of a settings file instead of
    relying on your shell. ... ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN"

    Minimum-change merge: read existing JSON, preserve all other keys, only
    set/overwrite `env.ANTHROPIC_BASE_URL` and `env.ANTHROPIC_AUTH_TOKEN`.
    Backup the file before writing.
    """
    home = Path(os.path.expanduser("~"))
    settings = home / ".claude" / "settings.json"
    backup = _backup(settings)
    existing = _read_json(settings)
    merged = _deep_merge(existing, {
        "env": {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_AUTH_TOKEN": dummy_key,
        }
    })
    _write_json(settings, merged)
    msg = f"wrote {settings} (env.ANTHROPIC_BASE_URL + env.ANTHROPIC_AUTH_TOKEN)"
    if backup:
        msg += f" [backup: {backup.name}]"
    return True, msg


# ---- Codex ----------------------------------------------------------------
def _configure_codex(base_url: str, dummy_key: str) -> tuple[bool, str]:
    """Codex — point the built-in `openai` provider at Atlas via `openai_base_url`.

    Source: https://developers.openai.com/codex/config-advanced
    "If you just need to point the built-in OpenAI provider at an LLM proxy,
    router, or data-residency enabled project, set `openai_base_url` in
    `config.toml` instead of defining a new provider. This changes the base
    URL for the built-in `openai` provider without requiring a separate
    model_providers entry."

    The dummy API key is read from `OPENAI_API_KEY` shell env, which we tell
    the user to export (or we can add it to ~/.bashrc if they want us to).
    Minimum-change: only add `openai_base_url` if not already present.
    """
    home = Path(os.path.expanduser("~"))
    cfg = home / ".codex" / "config.toml"
    backup = _backup(cfg)

    atlas_url = f"{base_url}/v1"

    existing_text = cfg.read_text() if cfg.exists() else ""

    # Idempotent: if we've already added it (comment marker), skip
    if "Added by Atlas proxy installer" in existing_text and "openai_base_url" in existing_text:
        msg = f"{cfg} already configured for Atlas — leaving alone"
        if backup:
            msg += f" [backup: {backup.name}]"
        return True, msg

    new_text = (
        existing_text
        + f"\n# Added by Atlas proxy installer ({_timestamp()})\n"
        + f'openai_base_url = "{atlas_url}"\n'
        + "# Atlas uses its own key pool — set OPENAI_API_KEY in your shell to any\n"
        + "# non-empty value (e.g. `export OPENAI_API_KEY=sk-atlas-dummy`).\n"
    )
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(new_text)

    # Persist the dummy key in shell rc so codex always finds it
    for rc in (".bashrc", ".zshrc"):
        rc_path = home / rc
        if rc_path.exists():
            marker = "# Atlas proxy installer: dummy API key"
            content = rc_path.read_text()
            if "OPENAI_API_KEY=sk-atlas-dummy" in content:
                continue
            with rc_path.open("a") as fh:
                fh.write(f"\n{marker}\nexport OPENAI_API_KEY={dummy_key}\n")

    msg = f"wrote {cfg} (openai_base_url={atlas_url}) + shell rc (OPENAI_API_KEY)"
    if backup:
        msg += f" [backup: {backup.name}]"
    return True, msg


# ---- Hermes ---------------------------------------------------------------
def _configure_hermes(base_url: str, dummy_key: str) -> tuple[bool, str]:
    """Hermes — use the built-in `hermes config set` CLI.

    Source: `hermes config --help` shows {show,edit,get,set,unset,path,...}.
    `hermes config set` is the official safe way to update config; it does
    its own backup before writing (verified: ~/.hermes/config.yaml.bak.*).

    We set model.base_url + model.api_key + model.provider so Hermes routes
    through Atlas. We do NOT touch model.default — the user can pick their
    default model after setup via `hermes model`.
    """
    if shutil.which("hermes") is None:
        return False, "hermes not on PATH"

    commands = [
        ["hermes", "config", "set", "model.base_url", f"{base_url}/v1"],
        ["hermes", "config", "set", "model.provider", "custom"],
        ["hermes", "config", "set", "model.api_key", dummy_key],
    ]
    errors = []
    for cmd in commands:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            errors.append(f"{' '.join(cmd[1:3])}: {r.stderr.strip() or r.stdout.strip()}")

    if errors:
        return False, "; ".join(errors)
    return True, f"hermes config set model.base_url/provider/api_key (backups in {os.path.expanduser('~')}/.hermes/)"


# ---- Pi -------------------------------------------------------------------
def _configure_pi(base_url: str, dummy_key: str) -> tuple[bool, str]:
    """Pi — add an `atlas` custom provider to ~/.pi/agent/models.json.

    Source: https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md
    "Add Ollama, LM Studio, vLLM, or any provider that speaks a supported API
    ... in models.json. { "providers": { "name": { "baseUrl": ..., "api":
    "openai-completions", "apiKey": ... } } }"

    Minimum-change merge: read existing models.json, preserve all other
    providers/models, only add the `atlas` provider entry. We deliberately
    do NOT change `defaultProvider` in settings.json — Pi resolves
    `defaultProvider + defaultModel` as a joined ID like "provider/model",
    so changing one without the matching model would break existing setups.
    The user picks atlas interactively via /model after setup.

    We seed one placeholder model so the smoke test can run end-to-end
    without requiring the user to have picked one first. The proxy accepts
    ANY model name — it'll pass it through to the upstream provider. The
    user replaces this with their real default via /model.
    """
    home = Path(os.path.expanduser("~"))
    cfg = home / ".pi" / "agent" / "models.json"
    backup = _backup(cfg)

    existing = _read_json(cfg)
    providers = existing.get("providers", {})
    if "atlas" in providers:
        msg = f"{cfg} already has atlas provider — leaving alone"
        if backup:
            msg += f" [backup: {backup.name}]"
        return True, msg

    providers["atlas"] = {
        "name": "Atlas",
        "baseUrl": f"{base_url}/v1",
        "api": "openai-completions",
        "apiKey": dummy_key,
        "authHeader": True,
        "models": [
            # Placeholder — proxy passes any model name through to upstream.
            # User picks their real default via /model after setup.
            {"id": "atlas-default", "name": "Atlas (default)", "reasoning": False,
             "input": ["text"], "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
             "contextWindow": 128000, "maxTokens": 8192},
        ],
    }
    existing["providers"] = providers
    _write_json(cfg, existing)

    msg = f"wrote {cfg} (atlas provider + atlas-default model — pick your real default via /model)"
    if backup:
        msg += f" [backup: {backup.name}]"
    return True, msg


# ===========================================================================
# HARNESS SMOKE TEST — real one-shot where documented; --version fallback
# ===========================================================================

SMOKE_TEST_PROMPT = "Reply with the single word OK."


def _run_capture(bin_name: str, args: list[str], timeout: int) -> tuple[bool, str]:
    """Run a command, capture output, return (ok, summary)."""
    try:
        r = subprocess.run(
            [bin_name, *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return False, f"{bin_name} not found"
    except subprocess.TimeoutExpired:
        return False, f"{bin_name} timed out after {timeout}s"

    out = (r.stdout or "").strip()
    # Check stdout contains a sane response marker (not 401/403/error)
    out_lower = out.lower()
    if any(err in out_lower for err in ("401", "403", "unauthorized", "forbidden", "error")):
        return False, f"{bin_name} rejected auth: {out[:120]}"
    return r.returncode == 0, f"exit={r.returncode}, stdout={out[:120]}"


def _smoke_via_version_only(bin_name: str) -> tuple[bool, str, str]:
    """Fallback: just verify the binary exists and runs.

    Returns (ok, summary, caveat). Caveat is shown to the user so they know
    we did NOT verify the harness → Atlas round-trip.
    """
    if shutil.which(bin_name) is None:
        return False, f"{bin_name} not on PATH", "no fallback possible"
    r = subprocess.run([bin_name, "--version"], capture_output=True, text=True, timeout=20)
    ok = r.returncode == 0
    summary = f"{bin_name} --version exit={r.returncode}"
    caveat = "version check only — end-to-end not verified"
    return ok, summary, caveat


def _smoke_claude(_base_url: str, _dummy_key: str) -> tuple[bool, str, str]:
    """Claude Code — safe one-shot via `claude -p` in --bare mode.

    Source: https://code.claude.com/docs/en/headless
    "claude -p 'PROMPT' ... Query via SDK, then exit."
    --bare skips hooks/skills/MCP auto-load. --allowedTools "" denies all
    tools so Claude can't try to read files / run shell commands.
    """
    if shutil.which("claude") is None:
        return False, "claude not on PATH", "no fallback"
    ok, summary = _run_capture(
        "claude",
        ["--bare", "-p", SMOKE_TEST_PROMPT, "--allowedTools", "", "--output-format", "text"],
        timeout=60,
    )
    return ok, summary, ""


def _smoke_codex(_base_url: str, _dummy_key: str) -> tuple[bool, str, str]:
    """Codex — safe one-shot via `codex exec` with --ephemeral --sandbox read-only.

    Source: https://developers.openai.com/codex/cli/reference and
    https://www.codex-docs.com/en/docs/non-interactive-mode
    "Use codex exec ... for scripted or CI-style runs that should finish without
    human interaction."
    --ephemeral avoids retaining session files. --sandbox read-only is the
    safest setting for inspection-only tasks.

    We capture stderr (which codex uses for progress + errors) so we can
    surface config parse errors instead of just reporting an empty stdout.
    """
    if shutil.which("codex") is None:
        return False, "codex not on PATH", "no fallback"

    try:
        r = subprocess.run(
            ["codex", "exec", "--ephemeral", "--sandbox", "read-only",
             "--skip-git-repo-check", SMOKE_TEST_PROMPT],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, "codex exec timed out after 120s", ""

    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    # If codex failed because config.toml has bad TOML, surface that — the
    # user can fix it themselves without us touching unrelated sections.
    if "Error loading config.toml" in err or "Error loading config.toml" in out:
        first_err = (err or out).splitlines()[0]
        return False, f"codex config parse error: {first_err}", "fix ~/.codex/config.toml"
    if r.returncode == 0:
        return True, f"exit=0, stdout={out[:120]}", ""
    return False, f"exit={r.returncode}, stderr={err[:200] or out[:200]}", ""


def _smoke_pi(_base_url: str, _dummy_key: str) -> tuple[bool, str, str]:
    """Pi — safe one-shot via `pi -p` (print mode).

    Source: https://www.agentscli.com/foundations/cheatsheets/pi/
    "pi -p 'PROMPT' — Process a prompt non-interactively and exit."
    --no-session avoids retaining session files for the smoke run.
    --provider atlas --model atlas-default explicitly targets the entry our
    installer added so the smoke goes through Atlas, not the user's
    existing default provider.
    """
    if shutil.which("pi") is None:
        return False, "pi not on PATH", "no fallback"
    ok, summary = _run_capture(
        "pi",
        ["-p", SMOKE_TEST_PROMPT, "--no-session", "--tools", "",
         "--provider", "atlas", "--model", "atlas-default"],
        timeout=60,
    )
    return ok, summary, ""


def _smoke_hermes(_base_url: str, _dummy_key: str) -> tuple[bool, str, str]:
    """Hermes — no documented safe non-interactive one-shot CLI command.

    The CLI is interactive (`hermes chat`) and has no `exec` / `-p` mode
    that matches Claude Code / Codex / Pi. Fall back to --version + a clear
    caveat that end-to-end wasn't tested.
    """
    return _smoke_via_version_only("hermes")


# ===========================================================================
# HARNESS REGISTRY
# ===========================================================================
class Harness:
    def __init__(
        self, id, label, bin_name,
        install_linux, install_macos, install_windows,
        configure, smoke,
    ):
        self.id = id
        self.label = label
        self.bin = bin_name
        self.install_linux = install_linux
        self.install_macos = install_macos
        self.install_windows = install_windows
        self.configure = configure
        self.smoke = smoke


HARNESSES: list[Harness] = [
    Harness(
        id="claude", label="Claude Code", bin_name="claude",
        install_linux=_install_claude_code, install_macos=_install_claude_code,
        install_windows=lambda: False,
        configure=_configure_claude, smoke=_smoke_claude,
    ),
    Harness(
        id="codex", label="Codex", bin_name="codex",
        install_linux=_install_codex, install_macos=_install_codex,
        install_windows=lambda: False,
        configure=_configure_codex, smoke=_smoke_codex,
    ),
    Harness(
        id="hermes", label="Hermes", bin_name="hermes",
        install_linux=_install_hermes, install_macos=_install_hermes,
        install_windows=_install_hermes,
        configure=_configure_hermes, smoke=_smoke_hermes,
    ),
    Harness(
        id="pi", label="Pi", bin_name="pi",
        install_linux=_install_pi, install_macos=_install_pi,
        install_windows=lambda: False,
        configure=_configure_pi, smoke=_smoke_pi,
    ),
]


# ===========================================================================
# UI helpers
# ===========================================================================
def _panel(title: str, body: str = "") -> Panel:
    text = Text.from_markup(body) if body else Text("")
    return Panel(text, title=title, title_align="left", border_style="cyan", padding=(0, 1))


def _pick_harness() -> Harness | None:
    body = "\n".join(f"  {i} {h.label}" for i, h in enumerate(HARNESSES, 1)) + "\n  5 Other (install later)"
    CONSOLE.print(_panel("Welcome to Atlas — choose your coding harness", body))
    raw = Prompt.ask("Select", choices=[str(i) for i in range(1, 6)], default="1")
    if raw == "5":
        return None
    idx = int(raw) - 1
    return HARNESSES[idx]


PROVIDERS = [
    ("openrouter",   "OpenRouter",   "Recommended"),
    ("nvidia",       "NVIDIA",       "Reliable, but slower"),
    ("huggingface",  "Hugging Face", "Frontier models / limited free usage"),
]


def _pick_provider() -> str | None:
    lines = []
    for i, (_, label, hint) in enumerate(PROVIDERS, 1):
        suffix = _c(" (Recommended)") if "Recommended" in hint else _dim(f" ({hint})")
        lines.append(f"  {i} {label}{suffix}")
    lines.append("  4 Skip")
    CONSOLE.print(_panel("Choose your API provider", "\n".join(lines)))
    raw = Prompt.ask("Select", choices=["1", "2", "3", "4"], default="1")
    if raw == "4":
        return None
    return PROVIDERS[int(raw) - 1][0]


def _proxy_key_file() -> Path:
    sys.path.insert(0, str(REPO_ROOT))
    from proxy.config import KEY_FILE  # type: ignore
    return Path(KEY_FILE)


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        ln.strip() for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    }


def _append_keys(path: Path, new_keys: list[str]) -> int:
    existing = _existing_keys(path)
    to_add = [k for k in new_keys if k not in existing]
    if not to_add:
        return 0
    suffix = "\n"
    if path.exists():
        content = path.read_text()
        if content and not content.endswith("\n"):
            suffix = "\n" + suffix
    with path.open("a") as fh:
        fh.write(suffix + "\n".join(to_add) + "\n")
    return len(to_add)


def _add_keys_individually() -> int:
    added = 0
    while True:
        key = Prompt.ask("Add API key", default="").strip()
        if not key:
            break
        target = _proxy_key_file()
        n = _append_keys(target, [key])
        added += n
        CONSOLE.print(_g("✓ Key added") if n else _y("✓ Key already present"))
        if not Confirm.ask("Add another?", default=False):
            break
    return added


def _add_keys_from_file(path: Path) -> tuple[int, int, int, int]:
    if not path.exists():
        CONSOLE.print(_r(f"file not found: {path}"))
        return (0, 0, 0, 0)
    target = _proxy_key_file()
    existing_before = _existing_keys(target)
    candidates_raw = [
        ln.strip() for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    seen = set()
    candidates = []
    for k in candidates_raw:
        if k not in seen:
            seen.add(k)
            candidates.append(k)
    added_count = _append_keys(target, candidates)
    duplicates = len(candidates) - added_count
    return (len(existing_before), len(candidates), duplicates, added_count)


def _keys_already_present() -> bool:
    return bool(_existing_keys(_proxy_key_file()))


# ===========================================================================
# Proxy health
# ===========================================================================
def _proxy_listen_url() -> str:
    """Resolve the actual proxy URL.

    Priority:
      1. ``data/bound_port`` — written by the running proxy with the
         port it actually bound to (honors port-collision fallback).
      2. ``LISTEN_PORT`` env var (matches ``proxy.config`` default).
      3. The fork's default (8777, NOT prod's 8788).
    """
    host = os.environ.get("LISTEN_HOST", "0.0.0.0")
    port: int | None = None
    try:
        bp = REPO_ROOT / "data" / "bound_port"
        if bp.exists():
            v = int(bp.read_text().strip())
            if 1 <= v <= 65535:
                port = v
    except (ValueError, OSError):
        pass
    if port is None:
        env_port = os.environ.get("LISTEN_PORT")
        if env_port and env_port.isdigit():
            port = int(env_port)
    if port is None:
        port = 8777  # fork default; prod lives on 8788
    return f"http://{'127.0.0.1' if host in ('0.0.0.0',) else host}:{port}"


def _proxy_is_up() -> bool:
    import urllib.request
    url = _proxy_listen_url()
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        # Runtime installation should already have started the proxy.
        # If it did not come up, use the existing run.sh background fallback.
        try:
            subprocess.run(
                ["bash", str(REPO_ROOT / "run.sh"), "--bg"],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except Exception:
            return False

        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False


# ===========================================================================
# Final verification
# ===========================================================================
def _verify_all(harness: Harness | None, dummy_key: str) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    base_url = _proxy_listen_url()

    # 1. Proxy installed
    results.append(("Proxy installed", VENV_PY.exists(), str(VENV_PY)))
    # 2. Proxy running
    up = _proxy_is_up()
    results.append(("Proxy running", up, _proxy_listen_url() + "/health"))
    # 3. Keys configured
    keys_ok = bool(_existing_keys(_proxy_key_file()))
    results.append(("API keys configured", keys_ok, str(_proxy_key_file())))

    if not harness:
        results.append(("Harness installed", True, "skipped (Other)"))
        results.append(("Harness configured", True, "skipped (Other)"))
        results.append(("Atlas connection test", True, "skipped (no harness)"))
        return results

    installed = shutil.which(harness.bin) is not None
    results.append((f"Harness installed ({harness.label})", installed, harness.bin))
    cfg_ok, cfg_msg = harness.configure(base_url, dummy_key)
    results.append((f"Harness configured ({harness.label})", cfg_ok, cfg_msg))
    if up and installed:
        smoke_ok, smoke_msg, caveat = harness.smoke(base_url, dummy_key)
        detail = smoke_msg
        if caveat:
            detail = f"{smoke_msg}  [{caveat}]"
        results.append(("Atlas connection test", smoke_ok, detail))
    else:
        results.append(("Atlas connection test", False, "skipped (proxy or harness missing)"))
    return results


# ===========================================================================
# Launch the harness
# ===========================================================================
def _launch_harness(harness: Harness) -> None:
    if not sys.stdin.isatty():
        CONSOLE.print(_g("✓ Setup complete"))
        CONSOLE.print("")
        CONSOLE.print("Atlas is ready.")
        CONSOLE.print(_c(f"Run: {harness.bin}"))
        return
    CONSOLE.print(_g("✓ Setup complete"))
    CONSOLE.print("")
    CONSOLE.print(f"Launching {harness.label}...")
    bin_path = shutil.which(harness.bin) or harness.bin
    try:
        os.execvp(bin_path, [bin_path])
    except FileNotFoundError:
        CONSOLE.print(_r(f"failed to exec {bin_path} — run it manually"))


# ===========================================================================
# Main wizard
# ===========================================================================
def run_wizard(dry_run: bool = False) -> int:
    title = "Atlas proxy — install (DRY-RUN)" if dry_run else "Atlas proxy — install"
    CONSOLE.print(Panel(_c(title), border_style="cyan"))

    # ---- 1. Bootstrap proxy ----
    if not bootstrap_proxy(dry_run):
        return 1

    # ---- 2. Pick harness ----
    harness = _pick_harness()
    if harness is None:
        CONSOLE.print(_y("No harness selected — finishing with proxy install only."))
        return 0

    # ---- 3. Detect/install harness ----
    already = shutil.which(harness.bin)
    harness_ready = bool(already)
    claude_install_state = None

    if already:
        CONSOLE.print(_g(f"✓ {harness.label} detected at {already}"))

    elif dry_run:
        CONSOLE.print(
            _dim(f"[harness] dry-run — would install {harness.label}")
        )

    elif harness.id == "claude" and detect_os() in ("linux", "macos"):
        # Claude installation runs independently while the rest of Atlas
        # setup continues. It is NOT configured until final verification.
        claude_install_state = _start_claude_code_install()

        if claude_install_state is not None:
            CONSOLE.print(
                _g("✓ Claude Code installation started in background")
            )
        else:
            CONSOLE.print(
                _y("⚠ Claude Code installation could not be started")
            )

    else:
        with Live(
            Spinner("dots", text=Text("")),
            console=CONSOLE,
            transient=True,
        ) as live:
            live.update(_c(f"Installing {harness.label}..."))

            os_name = detect_os()
            installer = {
                "linux": harness.install_linux,
                "macos": harness.install_macos,
                "windows": harness.install_windows,
            }.get(os_name, harness.install_linux)

            install_ok = installer()

            if install_ok and shutil.which(harness.bin):
                harness_ready = True
                live.update(
                    _g(f"✓ {harness.label} installed and verified")
                )
            else:
                live.update(
                    _r(f"✗ {harness.label} installation failed")
                )

    # Claude is deliberately NOT configured here.
    # Its installer is finalized near the end of the wizard.


    # ---- 4. Pick provider + keys ----
    provider = _pick_provider()
    if provider and not _keys_already_present():
        CONSOLE.print("")
        CONSOLE.print(_c("How would you like to add your API keys?"))
        CONSOLE.print("  1 Add keys individually")
        CONSOLE.print("  2 Use an existing key file")
        mode = Prompt.ask("Select", choices=["1", "2"], default="1")
        if mode == "1":
            _add_keys_individually()
        else:
            path_str = Prompt.ask("Path to key file")
            path = Path(os.path.expanduser(path_str))
            existing, imported, dups, added = _add_keys_from_file(path)
            CONSOLE.print(f"  {_c('Existing')}:    {existing}")
            CONSOLE.print(f"  {_c('Imported')}:    {imported}")
            CONSOLE.print(f"  {_c('Duplicates')}:  {dups}")
            CONSOLE.print(f"  {_c('Added')}:       {added}")
            CONSOLE.print(f"  {_c('Total')}:       {existing + added}")
    elif _keys_already_present():
        CONSOLE.print(_g("✓ Existing key file detected — skipping key setup"))

    # ---- 5. Configure harness ----
    base_url = _proxy_listen_url()
    dummy_key = "sk-atlas-dummy"  # proxy accepts any value; real keys are server-side
    if dry_run:
        CONSOLE.print(_dim(f"[configure] dry-run — would write {harness.label} config"))
    else:
        ok, msg = harness.configure(base_url, dummy_key)
        CONSOLE.print(_check(ok, f"Configured {harness.label}: {msg}"))

    # ---- Finalize Claude Code installation ----
    if harness.id == "claude" and claude_install_state is not None:
        with Live(
            Spinner("dots", text=Text("")),
            console=CONSOLE,
            transient=True,
        ) as live:
            live.update(_c("Finishing Claude Code installation..."))

            harness_ready = _finish_claude_code_install(
                claude_install_state
            )

            if harness_ready:
                live.update(_g("✓ Claude Code fully installed and verified"))
            else:
                live.update(_r("✗ Claude Code installation failed"))

    # Configure Claude ONLY after installation has completed and passed
    # executable verification.
    if harness.id == "claude" and harness_ready:
        base_url = _proxy_base_url()
        dummy_key = "atlas"

        try:
            harness.configure(base_url, dummy_key)
            CONSOLE.print(
                _g("✓ Claude Code configured for Atlas")
            )
        except Exception as exc:
            CONSOLE.print(
                _r(f"✗ Claude Code configuration failed: {exc}")
            )

    elif harness.id == "claude" and not harness_ready:
        CONSOLE.print(
            _y(
                "⚠ Claude Code is not fully installed — "
                "configuration skipped"
            )
        )

    # ---- 6. Verify everything ----
    CONSOLE.print("")
    CONSOLE.print(_c("Final verification"))
    results = _verify_all(harness, dummy_key)
    for label, ok, detail in results:
        CONSOLE.print("  " + _check(ok, f"{label}  {_dim(detail)}" if detail else label))

    failed = [r for r in results if not r[1]]
    if failed:
        CONSOLE.print("")
        CONSOLE.print(_y(f"⚠ {len(failed)} check(s) failed — review above"))

    # ---- 7. Launch ----
    if harness and not dry_run:
        _launch_harness(harness)
    elif harness:
        CONSOLE.print(_dim(f"[launch] dry-run — would exec {harness.bin}"))

    return 0 if not failed else 1
