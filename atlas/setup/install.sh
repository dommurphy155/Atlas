#!/usr/bin/env bash
# Atlas — bootstrap installer.
# Creates the venv, pip-installs requirements, then hands off to installer.py.
# This is the Atlas NVIDIA-only proxy (port 8788). It does not touch any other
# project or proxy.
set -euo pipefail

# Resolve PROJECT_DIR as the parent of this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color helpers — only when attached to a TTY.
if [[ -t 1 ]]; then
    CYAN=$'\033[36m'
    YELLOW=$'\033[33m'
    RED=$'\033[31m'
    RESET=$'\033[0m'
else
    CYAN=""
    YELLOW=""
    RED=""
    RESET=""
fi

info()  { printf '%s[info]%s %s\n' "$CYAN" "$RESET" "$*"; }
warn()  { printf '%s[warn]%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
error() { printf '%s[error]%s %s\n' "$RED" "$RESET" "$*" >&2; }

info "Atlas NVIDIA proxy installer"
info "Project dir: $PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# 1. Create venv if missing.
if [[ ! -x "$VENV_PYTHON" ]]; then
    info "Creating virtualenv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    info "Virtualenv already exists at $VENV_DIR"
fi

# 2. Upgrade pip and install requirements from the venv.
info "Upgrading pip ..."
"$VENV_PIP" install --upgrade pip

info "Installing requirements ..."
"$VENV_PIP" install -r "$PROJECT_DIR/setup/requirements.txt"

# 3. Hand off to the Python installer.
info "Handing off to Python installer ..."
exec "$VENV_PYTHON" "$PROJECT_DIR/setup/installer.py"
