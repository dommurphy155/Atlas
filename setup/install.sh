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
dry()  { printf '  \033[35m[DRY]\033[0m %s\n' "$*"; }

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
# ---------------------------------------------------------------------------
write_unit() {
    local unit_path="$1"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "write unit file: $unit_path (paths baked from $REPO_ROOT)"
        return 0
    fi
    cat >"$unit_path" <<EOF
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
WantedBy=default.target
EOF
    info "wrote $unit_path"
}

# ---------------------------------------------------------------------------
# systemd (system) install — root only
# ---------------------------------------------------------------------------
install_systemd_system() {
    local unit
    unit="$(system_unit_path)"

    if [[ -f "$unit" ]]; then
        if grep -q "Auto-generated by .*atlas_proxy/setup/install.sh" "$unit"; then
            info "refreshing existing unit (matches our marker)"
        else
            fail "$unit already exists and is NOT ours — refusing to overwrite.
  The existing unit is from a different atlas install. Either:
    - remove it manually (after stopping the service), OR
    - set ATLAS_SERVICE_NAME to a unique name (e.g. atlas-proxy-2)"
        fi
    fi
    write_unit "$unit"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "systemctl daemon-reload"
        dry "systemctl enable $SERVICE_NAME.service"
        ok "would enable $SERVICE_NAME.service (dry-run)"
        info "start with:   systemctl start $SERVICE_NAME"
        info "logs with:    journalctl -u $SERVICE_NAME -f"
    else
        systemctl daemon-reload
        systemctl enable "$SERVICE_NAME.service" >/dev/null
        ok "enabled $SERVICE_NAME.service"
        info "start with:   systemctl start $SERVICE_NAME"
        info "logs with:    journalctl -u $SERVICE_NAME -f"
    fi
}

# ---------------------------------------------------------------------------
# systemd (user) install — non-root with systemctl --user available
# ---------------------------------------------------------------------------
install_systemd_user() {
    local dir unit
    dir="$(user_unit_dir)"
    unit="$(user_unit_path)"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        dry "mkdir -p $dir"
        dry "write user unit: $unit"
        dry "systemctl --user daemon-reload"
        dry "systemctl --user enable $SERVICE_NAME.service"
        ok "would enable $SERVICE_NAME.service (user, dry-run)"
        info "start with:   systemctl --user start $SERVICE_NAME"
        info "logs with:    journalctl --user -u $SERVICE_NAME -f"
    else
        mkdir -p "$dir"
        # Only the user-scoped marker — avoids clobbering a hand-written unit.
        write_unit "$unit"
        systemctl --user daemon-reload
        systemctl --user enable "$SERVICE_NAME.service" >/dev/null
        ok "enabled $SERVICE_NAME.service (user)"
        info "start with:   systemctl --user start $SERVICE_NAME"
        info "logs with:    journalctl --user -u $SERVICE_NAME -f"
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

    ensure_dirs
    ensure_venv
    ensure_dotenv

    pick_runtime_mode "$requested"
    bold "Runtime: $RUNTIME_MODE"
    info "  reason: $RUNTIME_REASON"
    install_runtime

    install_cli_binary
    ok "install complete"
    info "  next: $BIN_NAME status"
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
