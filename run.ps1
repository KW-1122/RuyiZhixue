$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    python -m venv .venv
    & $Python -m pip install -r requirements.txt
}
& $Python scripts\diagnose.py
& $Python app.py
