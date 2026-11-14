$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

& $py -m unittest discover -s autoresearch/tests -v
& $py -m autoresearch.cli demo | Out-Host
Write-Host "Run-all finished."
