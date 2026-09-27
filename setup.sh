#!/usr/bin/env bash
# Insight Rx local setup (Linux / macOS). No GPU, cluster or internet access to model hubs is required.
#   bash setup.sh                 install, configure, and build the demo workspace from test_data/
#   bash setup.sh --no-demo-cases install and configure only
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'; then PY=$c; break; fi
  done
fi
[ -n "$PY" ] || { echo "Python 3.10, 3.11 or 3.12 is required (set PYTHON=/path/to/python)."; exit 1; }
echo "==> Python: $($PY --version)"
[ -d .venv ] || $PY -m venv .venv
. .venv/bin/activate
python -m pip install -q --upgrade pip
echo "==> Installing dependencies (CPU; a few minutes the first time)"
pip install -q -r requirements-local.txt
if [ ! -f .env ]; then
  SECRET=$(python -c 'import secrets; print(secrets.token_hex(24))')
  cat > .env <<ENV
# Local configuration written by setup.sh (edit freely; never commit)
INSIGHTRX_SECRET=$SECRET
INSIGHTRX_APP_ROOT=$(pwd)/local_data
INSIGHTRX_MODEL_DIR=$(pwd)/models/insightrx-onnx-v1
INSIGHTRX_AUTOSEED=1
# Optional integrations (leave empty to run fully offline; live data sources fall back to bundled snapshots)
# GEMINI_API_KEY=
# BACKBOARD_API_KEY=
# INSIGHTRX_EXTERNAL=offline
ENV
  echo "==> Wrote .env"
fi
set -a; . ./.env; set +a
mkdir -p "$INSIGHTRX_APP_ROOT"
echo "==> Verifying the model bundle"
python - <<'PYV'
import json, os, sys
d = os.environ["INSIGHTRX_MODEL_DIR"]
m = json.load(open(os.path.join(d, "manifest.json")))
missing = [f for f in m["sha256"] if not os.path.exists(os.path.join(d, f))]
if missing:
    sys.exit(f"model bundle incomplete, missing: {missing}")
print(f"   {len(m['sha256'])} files present; retina seeds {len(m['retina']['files'])}, encoders {len(m['encoders'])}")
PYV
if [ "${1:-}" != "--no-demo-cases" ]; then
  echo "==> Building the demo workspace from test_data/ (runs the real models on CPU; ~1 min per patient)"
  python scripts/seed_local.py
fi
echo
echo "Done. Start the app with:  bash run.sh     then open http://127.0.0.1:8000"
