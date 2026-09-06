#!/usr/bin/env bash
# Run the Atlas proxy from anywhere.
#
# Usage:
#   ./run.sh                 # foreground
#   ./run.sh --bg            # background, log to data/run.log
#   ./run.sh --stop          # kill background
#   ./run.sh --status        # check background
#   LISTEN_PORT=9000 ./run.sh
#
# The script resolves its own location, so it works no matter where you
# put the repo (~/atlas, ~/proxy, /opt/atlas_proxy, ...).

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root from this script's location
# ---------------------------------------------------------------------------
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Load .env if present (does not override already-exported vars)
# ---------------------------------------------------------------------------
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

# ---------------------------------------------------------------------------
# Pick venv / python — ALWAYS prefer repo-local venv so we don't pick up
# editable installs / site-packages from elsewhere.
# ---------------------------------------------------------------------------
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo "FATAL: no python3 found (need python3 on PATH to bootstrap venv)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Ensure repo-local venv exists + deps installed (idempotent).
# Skipping this and trusting the system python lets an editable install of
# `proxy` from another repo shadow ours — we MUST isolate.
# ---------------------------------------------------------------------------
if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
    echo "[run.sh] creating venv at $REPO_ROOT/.venv ..."
    "$PY" -m venv "$REPO_ROOT/.venv"
fi
PY="$REPO_ROOT/.venv/bin/python"

# Drop any inherited PYTHONPATH / sitecustomize hijacks that could shadow our
# package with a different `proxy/` from another repo on the same machine.
unset PYTHONPATH
PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE

need_install=0
if ! "$PY" -c "import fastapi, uvicorn, httpx, orjson" >/dev/null 2>&1; then
    need_install=1
fi
if [[ "$need_install" -eq 1 ]]; then
    echo "[run.sh] installing requirements into venv ..."
    "$PY" -m pip install -q --upgrade pip wheel
    "$PY" -m pip install -q -r "$REPO_ROOT/requirements.txt"
fi

# ---------------------------------------------------------------------------
# Ensure data/ dirs exist (config reads them on import)
# ---------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/data/openrouter_data" \
         "$REPO_ROOT/data/huggingface_data" \
         "$REPO_ROOT/data/proxy_data" \
         "$REPO_ROOT/data/payloads" \
         "$REPO_ROOT/proxy/logs"

# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------
PIDFILE="$REPO_ROOT/data/run.pid"
LOGFILE="$REPO_ROOT/data/run.log"

action="${1:-fg}"
case "$action" in
    --bg|bg)
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "[run.sh] already running (pid=$(cat "$PIDFILE"))"
            exit 0
        fi
        nohup "$PY" -m proxy.main >"$LOGFILE" 2>&1 &
        echo $! >"$PIDFILE"
        echo "[run.sh] started in background (pid=$(cat "$PIDFILE"), log=$LOGFILE)"
        ;;
    --stop|stop)
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            kill "$(cat "$PIDFILE")"
            rm -f "$PIDFILE"
            echo "[run.sh] stopped"
        else
            rm -f "$PIDFILE"
            echo "[run.sh] not running"
        fi
        ;;
    --status|status)
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "[run.sh] running (pid=$(cat "$PIDFILE"))"
        else
            echo "[run.sh] not running"
        fi
        ;;
    --help|-h|help)
        sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    fg|"")
        exec "$PY" -m proxy.main
        ;;
    *)
        echo "Unknown argument: $action (try --help)" >&2
        exit 2
        ;;
esac