#!/usr/bin/env python3
"""Atlas NVIDIA Proxy — Windows Service host.

A thin pywin32 wrapper that runs the Atlas proxy (`python -m proxy.atlas_proxy`)
as a managed Windows Service. The service spawns the proxy as a child process
and controls its lifecycle (start/stop). It does **not** import or modify any
`proxy.*` module — it only shells out to the proxy's own `__main__`, so the
proxy layer stays identical across Linux/macOS/Windows.

CLI (driven by setup/installer.py on Windows):
    atlas_windows_service.py --install      register + set auto-start
    atlas_windows_service.py --uninstall    unregister
    atlas_windows_service.py --start       start the service
    atlas_windows_service.py --stop        stop the service

pywin32 (`win32serviceutil` etc.) is imported lazily and only on Windows, so
this file compiles cleanly on Linux/macOS (where it's never executed) without
pywin32 installed.

Run as Administrator when installing/starting.
"""
from __future__ import annotations

import os
import sys
import platform
import subprocess
from pathlib import Path

# ── Path resolution (must work on every OS for import/compile) ─────────────
# PROJECT_DIR = atlas/  (parent of setup/). Venv lives at atlas/.venv.
PROJECT_DIR = Path(__file__).resolve().parent.parent
IS_WINDOWS = platform.system() == "Windows"

# The one portable command every service backend wraps.
_PROXY_CMD = ["-m", "proxy.atlas_proxy"]


def _venv_python() -> Path:
    """Venv interpreter: Scripts/python.exe on Windows, bin/python elsewhere.
    Only ever actually invoked on Windows, but resolves everywhere for safety."""
    scripts = PROJECT_DIR / ".venv" / ("Scripts" if IS_WINDOWS else "bin")
    return scripts / ("python.exe" if IS_WINDOWS else "python")


def _main_guarded() -> int:
    """On non-Windows, refuse politely — this module only makes sense there.
    Returns an exit code (never raises into the import path)."""
    if not IS_WINDOWS:
        sys.stderr.write(
            "atlas_windows_service.py: this host is %s, not Windows. "
            "The Windows Service backend is only used on Windows.\n"
            % platform.system()
        )
        return 1

    # Lazy import so the module compiles on Linux/macOS without pywin32.
    import servicemanager  # type: ignore
    import win32serviceutil  # type: ignore
    from pywin.framework import svcutils  # noqa: F401  (ensures win32 path)

    # Resolve the proxy launch command from the venv.
    proxy_python = str(_venv_python())

    class AtlasProxyService(win32serviceutil.ServiceFramework):
        _svc_name_ = "AtlasProxy"
        _svc_display_name_ = "Atlas NVIDIA Proxy"
        _svc_description_ = (
            "Atlas NVIDIA Proxy — routes Claude Code to NVIDIA's integrate API "
            "via a local OpenAI/Anthropic-compatible proxy on port 8788."
        )

        def __init__(self, args):
            super().__init__(args)
            self._proc: subprocess.Popen | None = None
            self._stop = False

        def SvcStop(self):
            """Stop the proxy child, then report SERVICE_STOPPED."""
            import win32service  # type: ignore
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            self._stop = True
            proc = self._proc
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            self.ReportServiceStatus(win32service.SERVICE_STOPPED)

        def SvcDoRun(self):
            """Spawn `<venv-python> -m proxy.atlas_proxy` and wait on it.
            The proxy's own resilience loop handles upstream failures; if the
            proxy exits, the service exits (Windows restarts it via auto-start
            on next boot, or the operator runs `atlas start`)."""
            import servicemanager as sm  # type: ignore
            sm.LogInfoMsg("AtlasProxy: starting proxy")
            try:
                self._proc = subprocess.Popen(
                    [proxy_python, *_PROXY_CMD],
                    cwd=str(PROJECT_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    # Inherit the service environment so ATLAS_* defaults apply.
                )
                self._proc.wait()
            except Exception as exc:  # pragma: no cover - service-runtime path
                sm.LogErrorMsg("AtlasProxy: proxy crashed: %s" % exc)
                raise

    # CLI dispatch: install/uninstall/start/stop via win32serviceutil helpers.
    if len(sys.argv) == 1:
        # No args + running as a service: let servicemanager handle it.
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(AtlasProxyService)
        servicemanager.StartServiceCtrlDispatcher()
        return 0

    arg = sys.argv[1].lower()
    if arg == "--install":
        win32serviceutil.InstallService(
            pythonClassString=AtlasProxyService.__module__ + ".AtlasProxyService",
            serviceName=AtlasProxyService._svc_name_,
            displayName=AtlasProxyService._svc_display_name_,
            description=AtlasProxyService._svc_description_,
            startType=2,  # SERVICE_AUTO_START — boot persistence
        )
        print("Installed service: %s" % AtlasProxyService._svc_name_)
        return 0
    if arg == "--uninstall":
        win32serviceutil.RemoveService(AtlasProxyService._svc_name_)
        print("Uninstalled service: %s" % AtlasProxyService._svc_name_)
        return 0
    if arg == "--start":
        win32serviceutil.StartService(AtlasProxyService._svc_name_)
        print("Started service: %s" % AtlasProxyService._svc_name_)
        return 0
    if arg == "--stop":
        win32serviceutil.StopService(AtlasProxyService._svc_name_)
        print("Stopped service: %s" % AtlasProxyService._svc_name_)
        return 0

    sys.stderr.write("Unknown argument: %s\n" % arg)
    sys.stderr.write(
        "Usage: atlas_windows_service.py "
        "[--install|--uninstall|--start|--stop]\n")
    return 2


if __name__ == "__main__":
    sys.exit(_main_guarded())
