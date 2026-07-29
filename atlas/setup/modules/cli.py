"""CLI binary installation."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .console import get_console
from .platform import PlatformInfo, detect_platform

console = get_console()

CLI_BASH = """#!/usr/bin/env bash
# atlas - operator CLI for the Atlas multi-provider proxy
# Controls atlas-proxy.service (systemd) on http://127.0.0.1:8788
set -euo pipefail

ATLAS_PROXY_URL="${ATLAS_PROXY_URL:-http://127.0.0.1:8788}"
ATLAS_SERVICE="${ATLAS_SERVICE:-atlas-proxy.service}"

# --- color setup (TTY-aware) ---
if [[ -t 1 ]]; then
  C_GREEN=$'\\033[32m'; C_CYAN=$'\\033[36m'; C_YELLOW=$'\\033[33m'
  C_RED=$'\\033[31m'; C_BOLD=$'\\033[1m'; C_RESET=$'\\033[0m'
else
  C_GREEN=""; C_CYAN=""; C_YELLOW=""; C_RED=""; C_BOLD=""; C_RESET=""
fi

err()  { printf '%s%s%s\\n' "$C_RED" "$*" "$C_RESET" >&2; }
info() { printf '%s%s%s\\n' "$C_CYAN" "$*" "$C_RESET"; }
ok()   { printf '%s%s%s\\n' "$C_GREEN" "$*" "$C_RESET"; }
warn() { printf '%s%s%s\\n' "$C_YELLOW" "$*" "$C_RESET"; }

usage() {
  cat <<EOF
${C_BOLD}atlas${C_RESET} - operator CLI for the Atlas multi-provider proxy

${C_BOLD}Usage:${C_RESET}
  atlas <command> [options]

${C_BOLD}Commands:${C_RESET}
  start     Start the atlas-proxy systemd service
  stop      Stop the atlas-proxy systemd service
  restart   Restart the atlas-proxy systemd service
  status    Show service status + proxy /health and /health/keys
  logs      Follow the atlas-proxy journal, pretty-printed (Ctrl-C to exit)
            Strips host/unit prefix, ANSI, provider/failovers; recolors
            status (200 green/4xx orange/5xx red), total (<30s green/
            30-120s orange/>120s red), tokens (fixed cyan). Raw logs stay
            plain text — colour is viewer-side only.
            Pass extra journalctl flags for a one-shot query:
              atlas logs -n 100            last 100 lines
              atlas logs --since '10 min ago'
              atlas logs --raw             vanilla journalctl (no transform)
              atlas logs -o json | jq .    structured (raw passthrough)
  keys      Show detailed per-key statistics from /health/keys

${C_BOLD}Options (for start/restart):${C_RESET}
  --nvidia       Use NVIDIA provider (default)
  --openrouter   Use OpenRouter provider

${C_BOLD}Environment:${C_RESET}
  ATLAS_PROXY_URL   Proxy base URL (default: http://127.0.0.1:8788)
  ATLAS_SERVICE     systemd unit name (default: atlas-proxy.service)
EOF
}

require_root() {
  if [[ $EUID -ne 0 ]]; then
    err "must run as root (try: sudo atlas $1)"
    return 1
  fi
}

require_systemctl() {
  if ! command -v systemctl >/dev/null 2>&1; then
    err "systemctl not found - systemd not available on this host"
    return 1
  fi
}

set_provider_env() {
  local provider="$1"
  require_systemctl || return 1
  systemctl set-environment ATLAS_PROVIDER="$provider"
}

do_service() {
  local action="$1"
  local provider="${2:-openrouter}"
  require_systemctl || return 1
  require_root "$action" || return 1

  if [[ "$action" == "start" || "$action" == "restart" ]]; then
    set_provider_env "$provider"
    info "atlas: $action $ATLAS_SERVICE (provider=$provider)"
  else
    info "atlas: $action $ATLAS_SERVICE"
  fi

  if systemctl "$action" "$ATLAS_SERVICE"; then
    ok "atlas: $action done"
  else
    err "atlas: $action failed"
    return 1
  fi
}

pprint_json() {
  local body="$1"
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "$body" | python3 -m json.tool 2>/dev/null || printf '%s\\n' "$body"
  else
    printf '%s\\n' "$body"
  fi
}

probe() {
  local label="$1" path="$2"
  printf '\\n%s--- %s %s%s\\n' "$C_BOLD" "$label" "$ATLAS_PROXY_URL$path" "$C_RESET"
  local body
  if body=$(curl -sS --max-time 3 -m 3 "${ATLAS_PROXY_URL}${path}" 2>/dev/null); then
    if [[ -z "$body" ]]; then
      warn "empty response from $path"
    else
      pprint_json "$body"
    fi
  else
    warn "proxy not responding on $path"
  fi
}

do_status() {
  require_systemctl || return 1
  info "atlas: systemctl status $ATLAS_SERVICE"
  systemctl status "$ATLAS_SERVICE" 2>&1 || warn "(service not active or not found)"
  probe "health" "/health"
  probe "keys"   "/health/keys"
  echo
}

_atlas_log_filter() {
  python3 -c '
import sys,json,datetime,re
RESET="\\033[0m"
GREEN="\\033[32m"; ORANGE="\\033[33m"; RED="\\033[31m"; CYAN="\\033[36m"
color = sys.stdout.isatty()
def paint(c,s): return f"{c}{s}{RESET}" if color else s
def status_color(s):
    try: n=int(s)
    except ValueError: return ""
    if n==200: return GREEN
    if 400<=n<500: return ORANGE
    if n>=500: return RED
    return ""
def total_color(v):
    try: f=float(v)
    except ValueError: return ""
    if f<30: return GREEN
    if f<=120: return ORANGE
    return RED
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    try: e=json.loads(line)
    except json.JSONDecodeError: print(line,flush=True); continue
    raw_ts=e.get("__REALTIME_TIMESTAMP",0)
    try: ts=datetime.datetime.fromtimestamp(int(raw_ts)/1e6).strftime("%b %d %H:%M:%S")
    except (TypeError,ValueError,OSError): ts=""
    m=e.get("MESSAGE","")
    if isinstance(m,list): m=bytes(m).decode("utf-8","replace")
    m=str(m)
    m=re.sub(r"\\x1b\\[[0-9;]*m","",m)
    m=re.sub(r"^\\d{2}:\\d{2}:\\d{2}\\s+","",m)
    m=m.replace(" provider=nvidia","").replace(" provider=openrouter","")
    m=re.sub(r" failovers=\\d+","",m)
    m=m.replace(" in_tokens="," in=").replace(" out_tokens="," out=")
    def c_status(mm):
        c=status_color(mm.group(1))
        return f"status={paint(c,mm.group(1))}" if c else mm.group(0)
    m=re.sub(r"status=(\\d+)",c_status,m)
    def c_total(mm):
        c=total_color(mm.group(1))
        return f"total={paint(c,mm.group(1))}s" if c else mm.group(0)
    m=re.sub(r"total=([0-9.]+)s",c_total,m)
    def c_tokens(mm): return f"tokens={paint(CYAN,mm.group(1))}"
    m=re.sub(r"tokens=([0-9?]+)",c_tokens,m)
    print(f"{ts} {m}" if ts else m,flush=True)
'
}

do_logs() {
  require_systemctl || return 1
  local raw=0
  local -a jargs=()
  for a in "$@"; do
    if [[ "$a" == "--raw" ]]; then raw=1; else jargs+=("$a"); fi
  done
  local want_raw=$raw
  for a in "${jargs[@]}"; do
    [[ "$a" == "-o"* || "$a" == "--output="* ]] && want_raw=1
  done
  if (( want_raw )); then
    if [[ ${#jargs[@]} -gt 0 ]]; then
      exec journalctl -u "$ATLAS_SERVICE" --no-pager "${jargs[@]}"
    fi
    exec journalctl -u "$ATLAS_SERVICE" -f --no-pager
  fi
  if [[ ${#jargs[@]} -gt 0 ]]; then
    info "atlas: journalctl $ATLAS_SERVICE ${jargs[*]}"
    journalctl -u "$ATLAS_SERVICE" --no-pager -o json "${jargs[@]}" | _atlas_log_filter
    return
  fi
  info "atlas: following journal for $ATLAS_SERVICE (Ctrl-C to exit)"
  journalctl -u "$ATLAS_SERVICE" -f --no-pager -o json | _atlas_log_filter
}

do_keys() {
  info "atlas: fetching key stats from $ATLAS_PROXY_URL/health/keys"
  local body
  if body=$(curl -sS --max-time 5 -m 5 "${ATLAS_PROXY_URL}/health/keys" 2>/dev/null); then
    if [[ -n "$body" ]]; then
      pprint_json "$body"
    else
      warn "empty response from /health/keys"
    fi
  else
    warn "proxy not responding on /health/keys"
  fi
}

main() {
  local cmd="${1:-}"
  shift || true

  local provider="openrouter"
  local -a remaining=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --nvidia)
        provider="nvidia"
        shift
        ;;
      --openrouter)
        provider="openrouter"
        shift
        ;;
      *)
        remaining+=("$1")
        shift
        ;;
    esac
  done

  case "$cmd" in
    start)   do_service start "$provider" ;;
    stop)    do_service stop ;;
    restart) do_service restart "$provider" ;;
    status)  do_status ;;
    logs)    do_logs "${remaining[@]}" ;;
    keys)    do_keys ;;
    -h|--help|"") usage ;;
    *)
      err "unknown command: $cmd"
      echo
      usage
      exit 2
      ;;
  esac
}

main "$@"
"""

CLI_WINDOWS = """@echo off
:: atlas.cmd - Windows CLI for Atlas Proxy
:: Requires: atlas-proxy.service installed via setup

set ATLAS_PROXY_URL=http://127.0.0.1:8788
set ATLAS_SERVICE=atlas-proxy

if "%1"=="" (
    echo Usage: atlas ^<command^> [options]
    echo Commands: start, stop, restart, status, logs, keys
    goto :eof
)

if "%1"=="status" (
    systemctl status %ATLAS_SERVICE% 2>nul
    curl -s %ATLAS_PROXY_URL%/health
    curl -s %ATLAS_PROXY_URL%/health/keys
    goto :eof
)

if "%1"=="start" (
    set ATLAS_PROVIDER=%2
    if "%ATLAS_PROVIDER%"=="" set ATLAS_PROVIDER=openrouter
    net start %ATLAS_SERVICE%
    goto :eof
)

if "%1"=="stop" (
    net stop %ATLAS_SERVICE%
    goto :eof
)

if "%1"=="restart" (
    set ATLAS_PROVIDER=%2
    if "%ATLAS_PROVIDER%"=="" set ATLAS_PROVIDER=openrouter
    net stop %ATLAS_SERVICE% && net start %ATLAS_SERVICE%
    goto :eof
)

if "%1"=="logs" (
    powershell -Command "Get-WinEvent -LogName 'Application' -ProviderName '%ATLAS_SERVICE%' -MaxEvents 50 | Format-Table TimeCreated, Message -AutoSize"
    goto :eof
)

if "%1"=="keys" (
    curl -s %ATLAS_PROXY_URL%/health/keys | python -m json.tool
    goto :eof
)

echo Unknown command: %1
"""


def install_cli_unix(platform_info: PlatformInfo, project_dir: Path) -> bool:
    """Install atlas CLI on Unix (Linux/macOS)."""
    cli_src = project_dir / "bin" / "atlas"
    cli_dst = platform_info.bin_dir / "atlas"

    try:
        # Ensure source exists
        cli_src.parent.mkdir(parents=True, exist_ok=True)
        cli_src.write_text(CLI_BASH)
        cli_src.chmod(0o755)

        # Remove existing
        if cli_dst.exists() or cli_dst.is_symlink():
            if os.geteuid() == 0:
                cli_dst.unlink()
            else:
                subprocess.run(["sudo", "rm", "-f", str(cli_dst)], check=True)

        # Symlink
        if os.geteuid() == 0:
            cli_dst.symlink_to(cli_src)
        else:
            subprocess.run(["sudo", "ln", "-sf", str(cli_src), str(cli_dst)], check=True)

        console.success(f"Installed CLI: {cli_dst} -> {cli_src}")
        return True
    except Exception as e:
        console.error(f"Failed to install CLI: {e}")
        return False


def install_cli_windows(platform_info: PlatformInfo, project_dir: Path) -> bool:
    """Install atlas CLI on Windows."""
    try:
        # Write .cmd shim
        cli_src = project_dir / "bin" / "atlas.cmd"
        cli_src.parent.mkdir(parents=True, exist_ok=True)
        cli_src.write_text(CLI_WINDOWS)

        # Add to user PATH via setx
        paths_to_add = [str(platform_info.venv_bin), str(project_dir / "bin")]
        current_path = os.environ.get("PATH", "")
        for p in paths_to_add:
            if p not in current_path:
                subprocess.run(["setx", "PATH", f"{p};{current_path}"], check=False)
                current_path = f"{p};{current_path}"

        console.success(f"Windows CLI shim: {cli_src}")
        console.info("Added to user PATH (restart shell to take effect)")
        return True
    except Exception as e:
        console.error(f"Failed to install Windows CLI: {e}")
        return False


def install_cli(platform_info: PlatformInfo, project_dir: Path) -> bool:
    """Install CLI for current platform."""
    if platform_info.is_windows:
        return install_cli_windows(platform_info, project_dir)
    else:
        return install_cli_unix(platform_info, project_dir)