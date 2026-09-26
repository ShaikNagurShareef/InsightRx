#!/bin/bash
# Serve the Insight Rx models from this GPU machine to the deployed web app (free):
#   1. vision worker (insightrx/vision_api.py) on 127.0.0.1:8100
#   2. Cloudflare quick tunnel (no account needed) -> public https://*.trycloudflare.com URL
#   3. register that URL with the web app every 5 min (the URL changes each time this script restarts)
# Needs in .env (or the environment): INSIGHTRX_VISION_KEY (same value as on Vercel) and INSIGHTRX_APP_URL.
#   bash scripts/run_vision_tunnel.sh
# Stop with Ctrl+C; the web app then falls back to SIMULATED mode and labels it.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
: "${INSIGHTRX_VISION_KEY:?set INSIGHTRX_VISION_KEY}" "${INSIGHTRX_APP_URL:?set INSIGHTRX_APP_URL (e.g. https://insightrx.vercel.app)}"
BIN=$HOME/.local/bin; mkdir -p "$BIN"
if [ ! -x "$BIN/cloudflared" ]; then
  curl -sL -o "$BIN/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$BIN/cloudflared"
fi
LOG=${TMPDIR:-/tmp}/insightrx_tunnel_$$.log
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} CUDA_VISIBLE_DEVICES=${GPU:-0}
PY=/data/users3/nshaik3/Projects/Oculomics/RetiLink/venv/bin
$PY/uvicorn insightrx.vision_api:app --host 127.0.0.1 --port 8100 --log-level warning &
W=$!
"$BIN/cloudflared" tunnel --no-autoupdate --url http://127.0.0.1:8100 > "$LOG" 2>&1 &
T=$!
trap 'kill $W $T 2>/dev/null; rm -f "$LOG"' EXIT
echo "loading models..."
until curl -sf -H "X-InsightRx-Key: $INSIGHTRX_VISION_KEY" http://127.0.0.1:8100/health >/dev/null; do sleep 3; done
until URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1) && [ -n "$URL" ]; do sleep 2; done
echo "vision worker live at $URL"
while true; do
  curl -s -X POST "$INSIGHTRX_APP_URL/api/vision/register" -H "Content-Type: application/json" \
       -H "X-InsightRx-Key: $INSIGHTRX_VISION_KEY" -d "{\"url\": \"$URL\"}" && echo
  sleep 300
done
