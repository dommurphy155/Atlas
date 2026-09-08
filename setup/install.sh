#!/usr/bin/env bash
# Atlas proxy installer.
#
# Usage:
#   ./setup/install.sh                # full install (auto-picks the best mode)
#   ./setup/install.sh --user         # user-mode (no system systemd, even if root)
#   ./setup/install.sh --uninstall    # remove unit + venv
#   ./setup/install.sh --check        # verify install is healthy
#   ./setup/install.sh --dry-run      # print what would happen, do nothing
#   ./setup/install.sh --dry-run --uninstall   # preview removal
#
# Repo-location-agnostic — resolves its own root, so it works whether
# the repo lives at ~/atlas_proxy, ~/proxy, or anywhere else.
#
# Mode selection (in priority order):
#   1. systemd (system)         — root + system systemd
#   2. systemd (user)           — non-root + systemctl --user available
#   3. tmux                     — non-root + tmux on PATH
#   4. nohup                    — POSIX nohup fallback, runs run.sh --bg
#   5. manual                   — nothing worked, user must run ./run.sh
#
# The install NEVER fails because of the runtime mode — if nothing else
# is available, it falls through to a nohup launch (or manual) so the
# proxy can still be started by the user.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null \
              || python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")"
SETUP_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SETUP_DIR/.." && pwd)"
SERVICE_NAME="${ATLAS_SERVICE_NAME:-atlas-proxy}"
SERVICE_USER="${ATLAS_SERVICE_USER:-root}"
PY_BIN="${PYTHON:-python3}"
# Binary name installed at /usr/local/bin/<BIN_NAME> (root) or
# ~/.local/bin/<BIN_NAME> (non-root).  Default 'atlas' for general installs;
# set ATLAS_BIN_NAME=atlas2 for the DM fork to avoid colliding with the
# upstream bundle's `atlas` command.
BIN_NAME="${ATLAS_BIN_NAME:-atlas}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '  \033[36m·\033[0m %s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }
fatal() { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }
dry()  { printf '  \033[35m[DRY]\033[0m %s\n' "$*"; }
# soft_fail: print + return 1 (so the fallback chain can keep going).
# Use this when a runtime failed but the installer as a whole should keep trying.
soft_fail() { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; return 1; }

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
is_root() { [[ ${EUID:-$(id -u)} -eq 0 ]]; }

has_systemctl_system() {
    command -v systemctl >/dev/null 2>&1 && \
    [[ -d /run/systemd/system ]] 2>/dev/null
}

has_systemctl_user() {
    command -v systemctl >/dev/null 2>&1 && \
    systemctl --user status >/dev/null 2>&1
}

has_tmux() {
    command -v tmux >/dev/null 2>&1
}

has_nohup() {
    command -v nohup >/dev/null 2>&1
}

# sudo-availability for the non-root + system-systemd path. We only check
# that the binary exists; whether the user can authenticate is decided at
# runtime when systemctl is actually invoked. Per TASK spec, do not gate on
# `sudo -n` (would falsely conclude sudo is unavailable when password auth
# would have worked).
has_sudo() {
    command -v sudo >/dev/null 2>&1
}

# True iff we can plausibly drive the SYSTEM systemd service from this
# invocation — either root directly, or a non-root user with sudo.
can_drive_systemd_system() {
    is_root || has_sudo
}

# ---------------------------------------------------------------------------
# Health verification — short bounded polling on the proxy health endpoint.
# Returns 0 if healthy, non-zero otherwise. Bounded so the installer does
# not sit around for minutes when the proxy fails to come up.
#
# Port resolution order (highest priority first):
#   1. data/bound_port — written by the running proxy with the port it
#      actually bound to. Honors the port-collision fallback at runtime.
#   2. proxy.config.LISTEN_PORT — the configured default (fork: 8777,
#      NOT prod's 8788).
#   3. 8777 — fork's compiled-in default.
# ---------------------------------------------------------------------------
HEALTH_PORT=8777
# (1) bound_port file (runtime truth — what the proxy actually bound to)
if [[ -f "$REPO_ROOT/data/bound_port" ]]; then
    _resolved_port="$(cat "$REPO_ROOT/data/bound_port" 2>/dev/null || true)"
    if [[ "$_resolved_port" =~ ^[0-9]+$ ]] && (( _resolved_port >= 1 && _resolved_port <= 65535 )); then
        HEALTH_PORT="$_resolved_port"
    fi
fi
# (2) proxy.config.LISTEN_PORT (only if bound_port wasn't set)
if [[ "$HEALTH_PORT" == "8777" ]] && [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    _resolved_port="$("$REPO_ROOT/.venv/bin/python" - <<PY 2>/dev/null || true
import os, sys
sys.path.insert(0, "$REPO_ROOT")
try:
    from proxy.config import LISTEN_PORT
    print(int(LISTEN_PORT))
except Exception:
    print(8777)
PY
)"
    if [[ "$_resolved_port" =~ ^[0-9]+$ ]]; then
        HEALTH_PORT="$_resolved_port"
    fi
elif [[ "$HEALTH_PORT" == "8777" ]] && command -v "$PY_BIN" >/dev/null 2>&1; then
    _resolved_port="$("$PY_BIN" - <<PY 2>/dev/null || true
import os, sys
sys.path.insert(0, "$REPO_ROOT")
try:
    from proxy.config import LISTEN_PORT
    print(int(LISTEN_PORT))
except Exception:
    print(8777)
PY
)"
    if [[ "$_resolved_port" =~ ^[0-9]+$ ]]; then
        HEALTH_PORT="$_resolved_port"
    fi
fi
HEALTH_URL="http://127.0.0.1:${HEALTH_PORT}/health"
HEALTH_TIMEOUT_S=6

check_health() {
    local deadline=$((SECONDS + HEALTH_TIMEOUT_S))
    while (( SECONDS < deadline )); do
        if command -v curl >/dev/null 2>&1; then
            if curl -fsS --max-time 1 "$HEALTH_URL" >/dev/null 2>&1; then
                ok "health check passed ($HEALTH_URL)"
                return 0
            fi
        else
            # Fallback: try python's urllib.
            if "$PY_BIN" -c "import urllib.request,sys; sys.exit(0 if 200 <= urllib.request.urlopen(sys.argv[1], timeout=0.5).status < 300 else 1)" "$HEALTH_URL" >/dev/null 2>&1; then
                ok "health check passed ($HEALTH_URL)"
                return 0
            fi
        fi
        sleep 0.25
    done
    warn "health check failed ($HEALTH_URL not responding within ${HEALTH_TIMEOUT_S}s)"
    return 1
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
system_unit_path() {
    echo "/etc/systemd/system/${SERVICE_NAME}.service"
}

user_unit_dir() {
    echo "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
}

user_unit_path() {
    echo "$(user_unit_dir)/${SERVICE_NAME}.service"
}

cli_dest() {
    if is_root; then
        echo "/usr/local/bin/${BIN_NAME}"
    else
        echo "${HOME}/.local/bin/${BIN_NAME}"
    fi
}

cli_dest_dir() {
    local d
    d="$(cli_dest)"
    echo "${d%/*}"
}

# ---------------------------------------------------------------------------
# Filesystem prep
# ---------------------------------------------------------------------------
ensure_dirs() {
    local dirs=(
        "$REPO_ROOT/data/openrouter_data"
        "$REPO_ROOT/data/huggingface_data"
        "$REPO_ROOT/data/proxy_data"
        "$REPO_ROOT/data/payloads"
        "$REPO_ROOT/proxy/logs"
    )
    if [[ "$DRY_RUN" -eq 1 ]]; then
        for d in "${dirs[@]}"; do dry "mkdir -p $d"; done
    else
        mkdir -p "${dirs[@]}"
    fi
    info "data/ tree ready"
}

ensure_venv() {
    if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
        bold "Creating venv ($REPO_ROOT/.venv)"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            dry "$PY_BIN -m venv $REPO_ROOT/.venv"
            dry "$REPO_ROOT/.venv/bin/python -m pip install -q --upgrade pip wheel"
            ok "venv created (dry-run)"
        else
            "$PY_BIN" -m venv "$REPO_ROOT/.venv"
            "$REPO_ROOT/.venv/bin/python" -m pip install -q --upgrade pip wheel
            ok "venv created"
        fi
    else
        info "venv already present"
    fi

    bold "Installing requirements"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "$REPO_ROOT/.venv/bin/python -m pip install -q -r $REPO_ROOT/requirements.txt"
        ok "requirements installed (dry-run)"
    else
        "$REPO_ROOT/.venv/bin/python" -m pip install -q -r "$REPO_ROOT/requirements.txt"
        ok "requirements installed"
    fi
}

ensure_dotenv() {
    if [[ ! -f "$REPO_ROOT/.env" && -f "$REPO_ROOT/.env.example" ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            dry "cp $REPO_ROOT/.env.example $REPO_ROOT/.env"
            warn "would create .env from .env.example — edit it to set keys"
        else
            cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
            warn "created .env from .env.example — edit it to set keys"
        fi
    else
        info ".env present (or .env.example missing — skipped)"
    fi
}

# ---------------------------------------------------------------------------
# systemd unit generation — paths baked from $REPO_ROOT so the unit is
# valid no matter where the repo lives.
#
# Writes the unit through `sudo cp` (not shell redirection) when running
# as a non-root user, so we never try to redirect into /etc/systemd/system
# without privileges. Returns 0 on success, 1 on failure — the caller MUST
# check the return code and abort its own flow accordingly.
# ---------------------------------------------------------------------------
write_unit() {
    local unit_path="$1"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "write unit file: $unit_path (paths baked from $REPO_ROOT)"
        return 0
    fi

    # Stage the unit content to a temp file in the repo (always writable
    # for the current user), then install it with elevated privileges.
    # Never use `cat >"$unit_path"` — that fails for non-root and bash's
    # `set -e` doesn't always trigger on the redirection error, leaving
    # the installer to print bogus success messages.
    local tmp
    tmp="$(mktemp "$REPO_ROOT/data/.${SERVICE_NAME}.unit.XXXXXX")" || {
        warn "could not create temp unit file"
        return 1
    }
    # shellcheck disable=SC2064  # we want $tmp expanded now, not later.
    trap "rm -f '$tmp'" RETURN

    if ! cat >"$tmp" <<EOF
# Auto-generated by $SETUP_DIR/install.sh — do not edit by hand.
# Re-run install.sh to refresh after repo moves.
[Unit]
Description=Atlas Translation Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$REPO_ROOT
EnvironmentFile=-$REPO_ROOT/.env
ExecStart=$REPO_ROOT/.venv/bin/python -m proxy.main
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF
    then
        warn "could not stage unit file at $tmp"
        return 1
    fi

    # Install to the protected location. `sudo cp` works whether we are
    # root (sudo is a no-op) or a non-root user with sudo auth.
    if is_root; then
        if ! install -m 644 "$tmp" "$unit_path" 2>/dev/null; then
            warn "install -m 644 $tmp $unit_path failed"
            return 1
        fi
    else
        if ! sudo install -m 644 "$tmp" "$unit_path" 2>/dev/null; then
            warn "sudo install -m 644 $tmp $unit_path failed (sudo auth or permission)"
            return 1
        fi
    fi

    info "wrote $unit_path"
    return 0
}

# ---------------------------------------------------------------------------
# systemd (system) install — root OR non-root with sudo.
# Privileged operations prefix `sudo` when not already root. We don't use
# `sudo -n`: per TASK spec, an interactive password prompt is acceptable.
# ---------------------------------------------------------------------------
install_systemd_system() {
    local unit
    unit="$(system_unit_path)"

    # Prepend sudo for non-root invocations of privileged commands.
    local _sudo=()
    is_root || _sudo=(sudo)

    if [[ -f "$unit" ]]; then
        if grep -q "Auto-generated by .*atlas_proxy/setup/install.sh" "$unit"; then
            info "refreshing existing unit (matches our marker)"
        else
            soft_fail "$unit already exists and is NOT ours — refusing to overwrite.
  The existing unit is from a different atlas install. Either:
    - remove it manually (after stopping the service), OR
    - set ATLAS_SERVICE_NAME to a unique name (e.g. atlas-proxy-2)"
        fi
    fi
    # write_unit uses sudo internally for non-root, so we don't prefix
    # sudo here. Any failure aborts the runtime attempt — never print
    # success messages below for ops that didn't actually succeed.
    if ! write_unit "$unit"; then
        soft_fail "could not write $unit — aborting systemd install"
        return 1
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "${_sudo[*]} systemctl daemon-reload"
        dry "${_sudo[*]} systemctl enable $SERVICE_NAME.service"
        dry "${_sudo[*]} systemctl start $SERVICE_NAME.service"
        ok "would enable + start $SERVICE_NAME.service (dry-run)"
        info "logs with:    journalctl -u $SERVICE_NAME -f"
        return 0
    else
        if ! "${_sudo[@]}" systemctl daemon-reload 2>/dev/null; then
            soft_fail "systemctl daemon-reload failed"
            return 1
        fi
        if ! "${_sudo[@]}" systemctl enable "$SERVICE_NAME.service" >/dev/null 2>&1; then
            soft_fail "systemctl enable failed"
            return 1
        fi
        ok "enabled $SERVICE_NAME.service"
        if ! "${_sudo[@]}" systemctl start "$SERVICE_NAME.service" 2>/dev/null; then
            soft_fail "systemctl start failed"
            return 1
        fi
        ok "started $SERVICE_NAME.service"
        info "logs with:    journalctl -u $SERVICE_NAME -f"
        return 0
    fi
}

# ---------------------------------------------------------------------------
# systemd (user) install — non-root with systemctl --user available
# ---------------------------------------------------------------------------
install_systemd_user() {
    local dir unit
    dir="$(user_unit_dir)"
    unit="$(user_unit_path)"

    # Validate that the user systemd bus is actually reachable
    # before attempting anything. Without XDG_RUNTIME_DIR or a
    # working loginctl session, systemctl --user will hang or
    # fail silently — and the installer must not print false
    # success messages.
    local xdg="${XDG_RUNTIME_DIR:-}"
    if [[ -z "$xdg" ]]; then
        xdg="/run/user/$(id -u)"
    fi
    if [[ ! -d "$xdg" ]]; then
        soft_fail "user systemd not available (no XDG_RUNTIME_DIR at $xdg)"
        return 1
    fi
    if ! systemctl --user is-active >/dev/null 2>&1 && \
       ! systemctl --user status >/dev/null 2>&1; then
        soft_fail "systemctl --user cannot communicate with the user bus"
        return 1
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "mkdir -p $dir"
        dry "write user unit: $unit"
        dry "systemctl --user daemon-reload"
        dry "systemctl --user enable $SERVICE_NAME.service"
        ok "would enable $SERVICE_NAME.service (user, dry-run)"
        info "start with:   systemctl --user start $SERVICE_NAME"
        info "logs with:    journalctl --user -u $SERVICE_NAME -f"
        return 0
    else
        mkdir -p "$dir"
        # Only the user-scoped marker — avoids clobbering a hand-written unit.
        if ! write_unit "$unit"; then
            soft_fail "could not write user unit — aborting"
            return 1
        fi
        if ! systemctl --user daemon-reload 2>/dev/null; then
            soft_fail "systemctl --user daemon-reload failed"
            return 1
        fi
        if ! systemctl --user enable "$SERVICE_NAME.service" >/dev/null 2>&1; then
            soft_fail "systemctl --user enable failed"
            return 1
        fi
        ok "enabled $SERVICE_NAME.service (user)"

        if ! systemctl --user start "$SERVICE_NAME.service" 2>/dev/null; then
            soft_fail "systemctl --user start failed"
            return 1
        fi
        ok "started $SERVICE_NAME.service (user)"

        info "logs with:    journalctl --user -u $SERVICE_NAME -f"
        return 0
    fi
}

# ---------------------------------------------------------------------------
# tmux install — non-root with tmux
# ---------------------------------------------------------------------------
install_tmux() {
    local session="atlas-proxy"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "tmux new-session -d -s $session \"$REPO_ROOT/run.sh\""
        ok "would launch tmux session: $session"
        info "attach with:   tmux attach -t $session"
        info "detach with:   Ctrl-b d"
    else
        # Kill any existing session first so we always have a clean start.
        tmux kill-session -t "$session" 2>/dev/null || true
        tmux new-session -d -s "$session" "$REPO_ROOT/run.sh"
        ok "launched tmux session: $session"
        info "attach with:   tmux attach -t $session"
        info "detach with:   Ctrl-b d"
    fi
}

# ---------------------------------------------------------------------------
# nohup install — POSIX fallback
# ---------------------------------------------------------------------------
install_nohup() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "$REPO_ROOT/run.sh --bg"
        ok "would launch in background via run.sh"
    else
        "$REPO_ROOT/run.sh" --bg
        ok "launched in background via run.sh"
    fi
    info "logs with:    tail -f $REPO_ROOT/data/run.log"
    info "stop with:    $REPO_ROOT/run.sh --stop"
}

# ---------------------------------------------------------------------------
# Runtime cleanup — best-effort teardown of a failed/runtime install.
# Never errors out; the caller has already moved on to the next fallback.
# ---------------------------------------------------------------------------
cleanup_runtime() {
    local mode="$1"
    case "$mode" in
        systemd)
            # Drive the system manager via sudo when non-root. `sudo`
            # in front of a command when already-root is a no-op so the
            # root path still works.
            if has_systemctl_system && can_drive_systemd_system; then
                local _sudo=()
                is_root || _sudo=(sudo)
                "${_sudo[@]}" systemctl disable --now "$SERVICE_NAME.service" 2>/dev/null || true
                "${_sudo[@]}" rm -f "$(system_unit_path)" 2>/dev/null || rm -f "$(system_unit_path)"
                "${_sudo[@]}" systemctl daemon-reload 2>/dev/null || true
            fi
            ;;
        systemd-user)
            if has_systemctl_user; then
                systemctl --user disable --now "$SERVICE_NAME.service" 2>/dev/null || true
                rm -f "$(user_unit_path)"
                systemctl --user daemon-reload 2>/dev/null || true
            fi
            ;;
        tmux)
            if has_tmux; then
                tmux kill-session -t atlas-proxy 2>/dev/null || true
            fi
            ;;
        nohup)
            if [[ -x "$REPO_ROOT/run.sh" ]]; then
                "$REPO_ROOT/run.sh" --stop 2>/dev/null || true
            fi
            rm -f "$REPO_ROOT/data/run.pid"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Auto-pick the best runtime mode for this host.
#
# Returns the mode name via $RUNTIME_MODE and writes a one-line summary
# to $RUNTIME_REASON so the caller can print it.
# ---------------------------------------------------------------------------
RUNTIME_MODE=""
RUNTIME_REASON=""

pick_runtime_mode() {
    local requested="${1:-auto}"

    if [[ "$requested" == "user" ]]; then
        # --user flag: prefer user-scope systemd, fall through to tmux/nohup
        if has_systemctl_user; then
            RUNTIME_MODE="systemd-user"
            RUNTIME_REASON="user-mode requested; systemctl --user available"
            return
        fi
        if has_tmux; then
            RUNTIME_MODE="tmux"
            RUNTIME_REASON="user-mode requested; no systemd-user; tmux available"
            return
        fi
        if has_nohup; then
            RUNTIME_MODE="nohup"
            RUNTIME_REASON="user-mode requested; no systemd-user or tmux; nohup available"
            return
        fi
        RUNTIME_MODE="manual"
        RUNTIME_REASON="user-mode requested; no runtime auto-launch available"
        return
    fi

    # auto
    if is_root && has_systemctl_system; then
        RUNTIME_MODE="systemd"
        RUNTIME_REASON="root + system systemd available"
        return
    fi
    if has_systemctl_user; then
        RUNTIME_MODE="systemd-user"
        RUNTIME_REASON="non-root + systemctl --user available"
        return
    fi
    if has_tmux; then
        RUNTIME_MODE="tmux"
        RUNTIME_REASON="no systemd; tmux available"
        return
    fi
    if has_nohup; then
        RUNTIME_MODE="nohup"
        RUNTIME_REASON="no systemd or tmux; nohup available"
        return
    fi
    RUNTIME_MODE="manual"
    RUNTIME_REASON="no auto-launch runtime found; start the proxy manually"
}

# ---------------------------------------------------------------------------
# Install dispatcher for the chosen mode
# ---------------------------------------------------------------------------
install_runtime() {
    local mode="$RUNTIME_MODE"
    case "$mode" in
        systemd)      install_systemd_system ;;
        systemd-user) install_systemd_user   ;;
        tmux)         install_tmux           ;;
        nohup)        install_nohup          ;;
        manual)       warn "no runtime manager available — start with: $REPO_ROOT/run.sh" ;;
        *)            fail "unknown runtime mode: $mode" ;;
    esac
}

# ---------------------------------------------------------------------------
# Fallback loop: try each candidate runtime in priority order. For each
# candidate, run the install/launch, verify the health endpoint, and if
# verification fails, clean up and move on. The first one that passes
# becomes RUNTIME_MODE. If everything fails we fall through to manual
# so the installer still completes successfully.
#
# Delegates to the Python `bin.runtime.install_runtime` so both this
# shell installer and `atlas install` (which routes through
# setup_wizard) share one runtime-selection code path. The wizard's
# interactive UX is preserved when invoked through `atlas install`;
# this shell entrypoint remains a non-interactive fallback for ops
# who just want the runtime installed.
# ---------------------------------------------------------------------------
_runtime_python() {
    local py="$REPO_ROOT/.venv/bin/python"
    if [[ ! -x "$py" ]]; then
        py="$PY_BIN"
    fi
    [[ -x "$(command -v "$py" 2>/dev/null)" ]] || {
        warn "no python interpreter available for runtime install"
        return 127
    }
    printf '%s\n' "$py"
}

try_runtime_with_fallback() {
    local requested="${1:-auto}"
    local py
    py="$(_runtime_python)" || {
        RUNTIME_MODE="manual"
        RUNTIME_REASON="no python interpreter for runtime install"
        return 0
    }

    # Drive the same runtime-selection logic the wizard uses. Stdout
    # is human-readable progress (relayed as-is), the final line is
    # "RUNTIME_RESULT=<mode>|<reason>" so we can parse the outcome.
    local output result_line mode reason
    output="$("$py" - "$REPO_ROOT" "$SERVICE_NAME" <<'PY' 2>&1
import os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "atlas"))
from bin import runtime as _rt
from pathlib import Path

repo = Path(sys.argv[1])
service = sys.argv[2]
venv_py = repo / ".venv" / "bin" / "python"

mode, ok, msg = _rt.install_runtime(repo, venv_py, service)
print(f"RUNTIME_RESULT={mode}|{msg}")
PY
)"
    result_line="$(printf '%s\n' "$output" | grep -E '^RUNTIME_RESULT=' | tail -n1 || true)"
    # Relay non-result lines as installer progress.
    printf '%s\n' "$output" | grep -v -E '^RUNTIME_RESULT=' | sed -E 's/^/  /'

    if [[ -z "$result_line" ]]; then
        RUNTIME_MODE="manual"
        RUNTIME_REASON="runtime install crashed"
        warn "Python runtime installer produced no result line — see logs"
        return 0
    fi

    local body="${result_line#RUNTIME_RESULT=}"
    RUNTIME_MODE="${body%%|*}"
    RUNTIME_REASON="${body#*|}"
    return 0
}

# ---------------------------------------------------------------------------
# CLI binary install — symlinks atlas/bin/atlas to the right place.
# Refuses to clobber an existing non-symlink of the same name.
# ---------------------------------------------------------------------------
install_cli_binary() {
    local src="$REPO_ROOT/atlas/bin/atlas"
    [[ -x "$src" ]] || fail "CLI script not found at $src"

    local dest
    dest="$(cli_dest)"
    local dest_dir
    dest_dir="$(cli_dest_dir)"

    # If a file already exists at $dest and is NOT our symlink, refuse to
    # clobber it — the operator probably has their own command with that name.
    if [[ -e "$dest" && ! -L "$dest" ]]; then
        warn "$dest exists and is not a symlink — leaving it alone"
        warn "  remove it manually or set ATLAS_BIN_NAME to something else"
        return 0
    fi
    # If the symlink points somewhere wrong, replace it.
    if [[ -L "$dest" ]]; then
        local current
        current="$(readlink -f "$dest" 2>/dev/null || true)"
        if [[ "$current" != "$(readlink -f "$src")" ]]; then
            if [[ "$DRY_RUN" -eq 1 ]]; then
                dry "rm $dest (existing symlink points elsewhere)"
            else
                rm -f "$dest"
            fi
        fi
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "mkdir -p $dest_dir"
        dry "ln -sf $src $dest"
        ok "would install CLI: $dest → $src"
    else
        mkdir -p "$dest_dir"
        ln -sf "$src" "$dest"
        ok "installed CLI: $dest → $src"
    fi
    info "use: $BIN_NAME status"
    if ! is_root && [[ ":$PATH:" != *":$dest_dir:"* ]]; then
        warn "  $dest_dir is not on your PATH — add it:"
        warn "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
}

uninstall_cli_binary() {
    local dest
    dest="$(cli_dest)"
    if [[ -L "$dest" ]]; then
        local current
        current="$(readlink "$dest" 2>/dev/null || true)"
        if [[ "$current" == "$REPO_ROOT/atlas/bin/atlas" ]]; then
            if [[ "$DRY_RUN" -eq 1 ]]; then
                dry "rm $dest"
                ok "would remove CLI symlink: $dest"
            else
                rm -f "$dest"
                ok "removed CLI symlink: $dest"
            fi
            return 0
        fi
    fi
    info "CLI symlink $dest not present (or not ours)"
}

# ---------------------------------------------------------------------------
# Top-level: install / uninstall / check
# ---------------------------------------------------------------------------
do_install() {
    local requested="${1:-auto}"
    bold "Atlas proxy installer"
    info "repo: $REPO_ROOT"
    info "service name: $SERVICE_NAME"
    info "user: $(id -un) (euid=${EUID:-?})"
    [[ "$DRY_RUN" -eq 1 ]] && warn "DRY-RUN — no changes will be made"

    # Bash bootstrap: data dirs, venv, .env. The runtime install + every
    # step after this point is delegated to the Python wizard so the user
    # gets the clean interactive setup (same code path as `atlas install`).
    ensure_dirs
    ensure_venv
    ensure_dotenv

    local atlas_py="$REPO_ROOT/atlas/bin/atlas"
    if [[ ! -x "$atlas_py" ]]; then
        fail "atlas CLI not found at $atlas_py"
    fi

    local py="$REPO_ROOT/.venv/bin/python"
    [[ -x "$py" ]] || py="$PY_BIN"

    local -a flags=("install")
    if [[ "$DRY_RUN" -eq 1 ]]; then
        flags+=("--dry-run")
    fi
    # The wizard auto-picks the best runtime and reads
    # ATLAS_SERVICE_NAME / ATLAS_BIN_NAME from the env (set at the top of
    # this script) so the shell installer and the Python wizard agree on
    # unit name + CLI symlink target. Note: the wizard does not currently
    # support a forced user-mode flag — it picks the best available mode.
    "$py" "$atlas_py" "${flags[@]}"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        warn "wizard exited with code $rc — install may be incomplete"
    fi
    # Echo a final line so callers parsing our output have a marker.
    echo "INSTALL_RC=$rc"
}

do_uninstall() {
    bold "Uninstalling"
    [[ "$DRY_RUN" -eq 1 ]] && warn "DRY-RUN — no changes will be made"

    # Try every possible runtime location — we don't know which one the
    # user picked at install time.
    local sys_unit user_unit
    sys_unit="$(system_unit_path)"
    user_unit="$(user_unit_path)"
    for unit in "$sys_unit" "$user_unit"; do
        if [[ -f "$unit" ]]; then
            if [[ "$unit" == "$user_unit" ]] && has_systemctl_user; then
                if [[ "$DRY_RUN" -eq 1 ]]; then
                    dry "systemctl --user disable --now $SERVICE_NAME.service"
                    dry "rm $unit"
                    dry "systemctl --user daemon-reload"
                    ok "would remove user systemd unit"
                else
                    systemctl --user disable --now "$SERVICE_NAME.service" 2>/dev/null || true
                    rm -f "$unit"
                    systemctl --user daemon-reload
                    ok "removed user systemd unit"
                fi
            elif [[ "$unit" == "$sys_unit" ]] && is_root && has_systemctl_system; then
                if [[ "$DRY_RUN" -eq 1 ]]; then
                    dry "systemctl disable --now $SERVICE_NAME.service"
                    dry "rm $unit"
                    dry "systemctl daemon-reload"
                    ok "would remove systemd unit"
                else
                    systemctl disable --now "$SERVICE_NAME.service" 2>/dev/null || true
                    rm -f "$unit"
                    systemctl daemon-reload
                    ok "removed systemd unit"
                fi
            else
                # Fall through — file exists but we don't have permission
                # to touch the manager.  Just remove the file.
                if [[ "$DRY_RUN" -eq 1 ]]; then
                    dry "rm $unit (no manager access)"
                else
                    rm -f "$unit"
                fi
            fi
        fi
    done
    # Kill tmux session if it exists.
    if has_tmux; then
        if tmux has-session -t atlas-proxy 2>/dev/null; then
            if [[ "$DRY_RUN" -eq 1 ]]; then
                dry "tmux kill-session -t atlas-proxy"
            else
                tmux kill-session -t atlas-proxy 2>/dev/null || true
            fi
        fi
    fi
    # Kill the nohup-style run.sh background if present.
    if [[ -f "$REPO_ROOT/run.sh" ]]; then
        "$REPO_ROOT/run.sh" --stop 2>/dev/null || true
    fi

    uninstall_cli_binary
    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "rm -f $REPO_ROOT/data/run.pid"
    else
        rm -f "$REPO_ROOT/data/run.pid"
    fi
    ok "uninstall complete (venv + data preserved)"
}

do_check() {
    bold "Health check"
    local rc=0
    info "repo: $REPO_ROOT"

    # Path resolution
    "$PY_BIN" - "$REPO_ROOT" <<'PY' || rc=1
import os, sys
os.chdir(sys.argv[1])
sys.path.insert(0, ".")
from proxy.config import (
    LISTEN_HOST, LISTEN_PORT,
    KEY_FILE, HF_KEY_FILE, RUNTIME_PROVIDER_FILE,
    SYSTEM_PROMPT_OVERRIDE_FILE, PAYLOAD_DIR, DATA_DIR,
)
print(f"  LISTEN       {LISTEN_HOST}:{LISTEN_PORT}")
print(f"  DATA_DIR     {DATA_DIR}")
print(f"  KEY_FILE     {KEY_FILE}")
print(f"  HF_KEY_FILE  {HF_KEY_FILE}")
print(f"  RUNTIME      {RUNTIME_PROVIDER_FILE}")
print(f"  PROMPT       {SYSTEM_PROMPT_OVERRIDE_FILE}")
print(f"  PAYLOAD_DIR  {PAYLOAD_DIR}")
PY

    # Required dirs
    for d in data/openrouter_data data/huggingface_data data/proxy_data \
             data/payloads proxy/logs; do
        if [[ -d "$REPO_ROOT/$d" ]]; then ok "$d/"; else warn "missing $d/"; rc=1; fi
    done

    # Venv + deps
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        ok "venv present"
        if "$REPO_ROOT/.venv/bin/python" -c "import fastapi, uvicorn, httpx, orjson" 2>/dev/null; then
            ok "deps importable"
        else
            warn "deps missing in venv"
            rc=1
        fi
    else
        warn "no venv"
        rc=1
    fi

    # Systemd units
    local sys_unit user_unit
    sys_unit="$(system_unit_path)"
    user_unit="$(user_unit_path)"
    if [[ -f "$sys_unit" ]]; then
        ok "systemd unit (system): $sys_unit"
    fi
    if [[ -f "$user_unit" ]]; then
        ok "systemd unit (user): $user_unit"
    fi
    if [[ ! -f "$sys_unit" && ! -f "$user_unit" ]]; then
        info "no systemd unit (tmux/nohup/manual?)"
    fi

    # CLI binary
    local bin_dest
    bin_dest="$(cli_dest)"
    if [[ -L "$bin_dest" ]]; then
        local target
        target="$(readlink "$bin_dest" 2>/dev/null || true)"
        if [[ "$target" == "$REPO_ROOT/atlas/bin/atlas" ]]; then
            ok "CLI symlink: $bin_dest → $target"
        else
            warn "CLI symlink $bin_dest points elsewhere: $target"
        fi
    elif [[ -e "$bin_dest" ]]; then
        warn "$bin_dest exists but is not our symlink"
    else
        info "no CLI symlink ($bin_dest missing)"
    fi

    if [[ $rc -eq 0 ]]; then ok "all green"; else warn "issues above"; fi
    return $rc
}

# DRY_RUN must be visible inside the python heredoc in do_check.
export DRY_RUN
case "${1:-}" in
    "")             DRY_RUN=0; do_install auto ;;
    --user)         DRY_RUN=0; do_install user ;;
    --uninstall)    DRY_RUN=0; do_uninstall ;;
    --check)        DRY_RUN=0; do_check ;;
    --dry-run)      shift; DRY_RUN=1
                    case "${1:-}" in
                        "")             do_install auto ;;
                        --user)         do_install user ;;
                        --uninstall)    do_uninstall ;;
                        --check)        do_check ;;
                        *)              usage; exit 2 ;;
                    esac ;;
    -h|--help|help) usage ;;
    *)              usage; exit 2 ;;
esac
