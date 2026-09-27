#!/usr/bin/env bash
# Start Insight Rx locally on CPU: http://127.0.0.1:8000  (PORT=8080 bash run.sh to change the port)
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run 'bash setup.sh' first."; exit 1; }
. .venv/bin/activate
set -a; . ./.env; set +a
echo "Insight Rx on http://127.0.0.1:${PORT:-8000}  (first page load warms up the models on CPU, ~20-40 s)"
exec python -m uvicorn insightrx.app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
