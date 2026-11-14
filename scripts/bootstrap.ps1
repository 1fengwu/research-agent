$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path (Join-Path $root ".venv"))) {
    python -m venv .venv
}

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements-lock.txt
if (Test-Path (Join-Path $root "requirements-app.txt")) {
    & $py -m pip install -r requirements-app.txt
}

Write-Host "Bootstrap done. Python: $py"
