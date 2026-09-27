# Insight Rx local setup (Windows PowerShell). No GPU or cluster required.
#   powershell -ExecutionPolicy Bypass -File setup.ps1            (add -NoDemoCases to skip building demo cases)
param([switch]$NoDemoCases)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = $null
foreach ($v in "3.12", "3.11", "3.10") { try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break } } catch {} }
if (-not $py) { Write-Error "Python 3.10, 3.11 or 3.12 is required (install from python.org, include the 'py' launcher)." }
Write-Host "==> Python: $(& $py[0] $py[1] --version)"
if (-not (Test-Path .venv)) { & $py[0] $py[1] -m venv .venv }
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPy -m pip install -q --upgrade pip
Write-Host "==> Installing dependencies (CPU; a few minutes the first time)"
& $venvPy -m pip install -q -r requirements-local.txt
if (-not (Test-Path .env)) {
  $secret = & $venvPy -c "import secrets; print(secrets.token_hex(24))"
  $root = (Get-Location).Path
  @"
# Local configuration written by setup.ps1 (edit freely; never commit)
INSIGHTRX_SECRET=$secret
INSIGHTRX_APP_ROOT=$root\local_data
INSIGHTRX_MODEL_DIR=$root\models\insightrx-onnx-v1
INSIGHTRX_AUTOSEED=1
# GEMINI_API_KEY=
# BACKBOARD_API_KEY=
"@ | Set-Content -Encoding UTF8 .env
  Write-Host "==> Wrote .env"
}
Get-Content .env | Where-Object { $_ -match '^\s*[A-Z_]+=' } | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item -Path "env:$k" -Value $v }
New-Item -ItemType Directory -Force -Path $env:INSIGHTRX_APP_ROOT | Out-Null
if (-not $NoDemoCases) {
  Write-Host "==> Building the demo workspace from test_data (real models on CPU; ~1 min per patient)"
  & $venvPy scripts\seed_local.py
}
Write-Host "`nDone. Start the app with:  powershell -ExecutionPolicy Bypass -File run.ps1   then open http://127.0.0.1:8000"
