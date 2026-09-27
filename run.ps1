# Start Insight Rx locally on CPU (Windows): http://127.0.0.1:8000
Set-Location $PSScriptRoot
if (-not (Test-Path .venv)) { Write-Error "Run setup.ps1 first." }
Get-Content .env | Where-Object { $_ -match '^\s*[A-Z_]+=' } | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item -Path "env:$k" -Value $v }
$port = if ($env:PORT) { $env:PORT } else { "8000" }
Write-Host "Insight Rx on http://127.0.0.1:$port  (first page load warms up the models on CPU)"
& .\.venv\Scripts\python.exe -m uvicorn insightrx.app.main:app --host 127.0.0.1 --port $port
