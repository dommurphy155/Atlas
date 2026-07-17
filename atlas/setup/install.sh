#!/usr/bin/env bash
# Atlas — bootstrap installer.
# Creates the venv, pip-installs requirements, then hands off to installer.py.
# This is the Atlas NVIDIA-only proxy (port 8788). It does not touch any other
# project or proxy.
set -euo pipefail

# Resolve PROJECT_DIR as the parent of this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Terminal styling ──────────────────────────────────────────────────────
# Only emit escape codes when stdout is a TTY; otherwise plain text so logs
# stay clean when piped/redirected.
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    CYAN=$'\033[36m'
    GREEN=$'\033[32m'
    YELLOW=$'\033[33m'
    RED=$'\033[31m'
    MAGENTA=$'\033[35m'
    RESET=$'\033[0m'
else
    BOLD=""; DIM=""; CYAN=""; GREEN=""; YELLOW=""; RED=""; MAGENTA=""; RESET=""
fi

# ── Helpers ───────────────────────────────────────────────────────────────
hr() {
    # A thin rule line the width of the banner (matches the ── above).
    printf '%s─── Atlas ──────────────────────────────────────────────────%s\n' "$DIM" "$RESET"
}

banner() {
    printf '\n'
    hr
    printf '%s%sAtlas%s %sNVIDIA proxy installer%s\n' \
        "$BOLD$MAGENTA" "$RESET" "$DIM" "$RESET"
    printf '%s%sPort 8788 · NVIDIA-only · Claude Code front-end%s\n' "$DIM" "$RESET"
    hr
}

# status <ok|skip|fail|run> <label>
#   ok   → green  ✔
#   skip → dim    ⊘
#   fail → red    ✘
#   run  → cyan   ▸  (in-progress; caller overwrites with ok/skip/fail)
status() {
    local kind="$1" label="$2"
    case "$kind" in
        ok)   printf '%s✔%s  %s%s\n' "$GREEN" "$RESET" "$label" "$RESET" ;;
        skip) printf '%s⊘%s  %s%s\n' "$DIM" "$RESET" "$DIM$label$RESET" ;;
        fail) printf '%s✘%s  %s%s\n' "$RED" "$RESET" "$RED$label$RESET" >&2 ;;
        run)  printf '%s▸%s  %s%s ... ' "$CYAN" "$RESET" "$label" "$RESET" ;;
    esac
}

# Overwrite the trailing " ... " left by a `run` line with a final marker.
# Uses \r so it stays tidy in a TTY; harmless in a pipe (the markers just
# follow the label on the same logical line).
finish_run() {
    local kind="$1" label="$2"
    case "$kind" in
        ok)   printf '\r%s✔%s  %s%s\n' "$GREEN" "$RESET" "$label" "$RESET" ;;
        skip) printf '\r%s⊘%s  %s%s\n' "$DIM" "$RESET" "$DIM$label$RESET" ;;
        fail) printf '\r%s✘%s  %s%s\n' "$RED" "$RESET" "$RED$label$RESET" >&2 ;;
    esac
}

# ── Start ──────────────────────────────────────────────────────────────────
banner
printf '%s▸%s  %sProject%s   %s%s%s\n' "$CYAN" "$RESET" "$DIM" "$RESET" "$CYAN" "$PROJECT_DIR" "$RESET"

VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
PIP_LOG_UPGRADE="$(mktemp -t atlas-pip-upgrade.XXXXXX)"
PIP_LOG_INSTALL="$(mktemp -t atlas-pip-install.XXXXXX)"
trap 'rm -f "$PIP_LOG_UPGRADE" "$PIP_LOG_INSTALL"' EXIT

# ── 1. Virtualenv ──────────────────────────────────────────────────────────
if [[ ! -x "$VENV_PYTHON" ]]; then
    status run "Creating virtualenv"
    if python3 -m venv "$VENV_DIR"; then
        finish_run ok "Creating virtualenv"
    else
        finish_run fail "Creating virtualenv"
        error "python3 -m venv failed"
        exit 1
    fi
else
    status skip "Virtualenv already exists"
fi

# ── 2. pip upgrade + requirements ──────────────────────────────────────────
# Bury pip's download bars and resolver chatter in temp logs; surface them
# only on failure (indented). PIP_QUIET also trims the final "Successfully
# installed" wall that would otherwise clutter the step list.
export PIP_QUIET=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_COLOR=1

status run "Upgrading pip"
if "$VENV_PIP" install --upgrade pip >"$PIP_LOG_UPGRADE" 2>&1; then
    finish_run ok "Upgrading pip"
else
    finish_run fail "Upgrading pip"
    printf '%s    pip output:%s\n' "$DIM" "$RESET" >&2
    sed 's/^/      /' "$PIP_LOG_UPGRADE" >&2 || true
    exit 1
fi

status run "Installing requirements"
if "$VENV_PIP" install -r "$PROJECT_DIR/setup/requirements.txt" >"$PIP_LOG_INSTALL" 2>&1; then
    finish_run ok "Installing requirements"
else
    finish_run fail "Installing requirements"
    printf '%s    pip output:%s\n' "$DIM" "$RESET" >&2
    sed 's/^/      /' "$PIP_LOG_INSTALL" >&2 || true
    exit 1
fi

# ── 3. Hand off to the Python installer ────────────────────────────────────
printf '%s▸%s  %sHanding off to Python installer%s\n' "$CYAN" "$RESET" "$DIM" "$RESET"
exec "$VENV_PYTHON" "$PROJECT_DIR/setup/installer.py"
