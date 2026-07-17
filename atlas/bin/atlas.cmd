@echo off
REM atlas.cmd - operator CLI for the Atlas NVIDIA proxy (Windows)
REM
REM Windows counterpart to the bash `bin/atlas`. Drives the AtlasProxy Windows
REM Service via sc.exe and runs the token tracker through the venv python.
REM Independent of the bash script (which stays the Linux/macOS command).
REM
REM Service name: AtlasProxy  (registered by setup/atlas_windows_service.py)
REM Proxy URL:     http://127.0.0.1:8788

setlocal enableextensions enabledelayedexpansion

set "ATLAS_PROXY_URL=%ATLAS_PROXY_URL%"
if "%ATLAS_PROXY_URL%"=="" set "ATLAS_PROXY_URL=http://127.0.0.1:8788"
set "ATLAS_SERVICE=%ATLAS_SERVICE%"
if "%ATLAS_SERVICE%"=="" set "ATLAS_SERVICE=AtlasProxy"

REM Resolve the repo root from this script's location (bin\atlas.cmd -> atlas\).
set "SCRIPT_DIR=%~dp0"
set "PROJ_DIR=%SCRIPT_DIR%.."
set "VENV_PY=%PROJ_DIR%\.venv\Scripts\python.exe"

if "%~1"=="" goto :usage
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--help" goto :usage

if /i "%~1"=="start"    goto :start
if /i "%~1"=="stop"     goto :stop
if /i "%~1"=="restart"  goto :restart
if /i "%~1"=="status"   goto :status
if /i "%~1"=="logs"     goto :logs
if /i "%~1"=="tokens"   goto :tokens

echo atlas: unknown command: %~1 1>&2
echo.
goto :usage

:start
echo atlas: starting %ATLAS_SERVICE%
sc start "%ATLAS_SERVICE%"
if errorlevel 1 (
  echo atlas: start failed ^(run as Administrator^) 1>&2
  exit /b 1
)
echo atlas: start done
goto :eof

:stop
echo atlas: stopping %ATLAS_SERVICE%
sc stop "%ATLAS_SERVICE%"
if errorlevel 1 (
  echo atlas: stop failed ^(run as Administrator^) 1>&2
  exit /b 1
)
echo atlas: stop done
goto :eof

:restart
echo atlas: restarting %ATLAS_SERVICE%
sc stop "%ATLAS_SERVICE%" >nul 2>&1
timeout /t 2 /nobreak >nul
sc start "%ATLAS_SERVICE%"
if errorlevel 1 (
  echo atlas: restart failed ^(run as Administrator^) 1>&2
  exit /b 1
)
echo atlas: restart done
goto :eof

:status
echo atlas: querying %ATLAS_SERVICE%
sc query "%ATLAS_SERVICE%"
echo.
echo --- health %ATLAS_PROXY_URL%/health ---
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri '%ATLAS_PROXY_URL%/health').Content } catch { Write-Host 'proxy not responding on /health' }"
echo.
echo --- stats %ATLAS_PROXY_URL%/stats ---
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri '%ATLAS_PROXY_URL%/stats').Content } catch { Write-Host 'proxy not responding on /stats' }"
goto :eof

:logs
REM The Windows Service host sends proxy stdout/stderr to DEVNULL (services
REM can't own a console). Tail the proxy's own log file if present, else hint.
set "LOG=%PROJ_DIR%\data\atlas-proxy.log"
if exist "%LOG%" (
  echo atlas: tailing %LOG% ^(Ctrl-C to exit^)
  powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Wait -Tail 100"
) else (
  echo atlas: no log file at %LOG%
  echo        The service host suppresses console output; enable ATLAS_PROXY_DEBUG=1
  echo        and redirect proxy output to a file to capture logs.
)
goto :eof

:tokens
if not exist "%VENV_PY%" (
  echo atlas: venv python not found at %VENV_PY% ^(re-run setup\install.ps1^) 1>&2
  exit /b 1
)
pushd "%PROJ_DIR%"
"%VENV_PY%" -m proxy.token_tracker
set "RC=%errorlevel%"
popd
exit /b %RC%

:usage
echo atlas - operator CLI for the Atlas NVIDIA proxy (Windows)
echo.
echo Usage:
echo   atlas ^<command^>
echo.
echo Commands:
echo   start     Start the AtlasProxy Windows Service
echo   stop      Stop the AtlasProxy Windows Service
echo   restart   Restart the AtlasProxy Windows Service
echo   status    Show service state + proxy /health and /stats
echo   logs      Tail the proxy log file (Ctrl-C to exit)
echo   tokens    Clean token/usage summary since last restart
echo   -h, --help, (none)   Show this help
echo.
echo Environment:
echo   ATLAS_PROXY_URL   Proxy base URL (default: http://127.0.0.1:8788)
echo   ATLAS_SERVICE     Windows Service name (default: AtlasProxy)
goto :eof
