#!/bin/bash
# Launch the RetiLink workspace locally (GPU inference if a trained model exists, else SIMULATED mode).
#   RETILINK_MODEL_DIR=/data/users3/nshaik3/Projects/Oculomics/RetiLink/1 GEMINI_API_KEY=... bash scripts/run_app.sh
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
[ -f .env ] && set -a && . ./.env && set +a
exec /data/users3/nshaik3/Projects/Oculomics/RetiLink/venv/bin/uvicorn retilink.app.main:app --host ${HOST:-127.0.0.1} --port ${PORT:-8000}
