# Atlas — Windows bootstrap installer.
# PowerShell mirror of setup/install.sh: create the venv, pip-install
# requirements, then hand off to the cross-platform installer.py (which
# detects Windows and runs the Windows-native flow: Windows Service, atlas.cmd
# shim, setx env wiring, PowerShell-based claude install).
#
# Usage (run as Administrator for service registration):
#   powershell -ExecutionPolicy Bypass -File setup\install.ps1
#
# This bootstrap contains NO install logic of its own — installer.py is the
# single source of truth for all OS-specific behavior.

$ErrorActionPreference = 'Stop'

# Resolve PROJECT_DIR as the parent of this script's directory.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

$VenvDir    = Join-Path $ProjectDir '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPip    = Join-Path $VenvDir 'Scripts\pip.exe'
$ReqFile    = Join-Path $ScriptDir 'requirements.txt'
$Installer  = Join-Path $ScriptDir 'installer.py'

function Write-Step($msg) { Write-Host "  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }

Write-Host ''
Write-Host '  Atlas NVIDIA proxy installer (Windows bootstrap)' -ForegroundColor Magenta
Write-Host "  Project: $ProjectDir" -ForegroundColor DarkGray
Write-Host ''

# 1. Virtualenv
if (-not (Test-Path $VenvPython)) {
    Write-Step 'Creating virtualenv'
    & python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Host '  [FAIL] python -m venv failed' -ForegroundColor Red; exit 1 }
    Write-Ok 'virtualenv created'
} else {
    Write-Ok 'virtualenv already exists'
}

# 2. pip upgrade + requirements (quiet; surface on failure)
Write-Step 'Upgrading pip'
& $VenvPip install --upgrade pip --quiet --disable-pip-version-check 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host '  [FAIL] pip upgrade failed' -ForegroundColor Red; exit 1 }
Write-Ok 'pip upgraded'

Write-Step 'Installing requirements'
& $VenvPip install -r $ReqFile --quiet --disable-pip-version-check 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host '  [FAIL] pip install -r failed' -ForegroundColor Red; exit 1 }
Write-Ok 'requirements installed'

# 3. Hand off to the Python installer (the single source of OS-aware logic).
Write-Step 'Handing off to Python installer'
& $VenvPython $Installer
exit $LASTEXITCODE
