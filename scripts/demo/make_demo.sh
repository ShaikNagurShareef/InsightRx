#!/bin/bash
# Regenerate the Insight Rx demo video: narration (Kokoro, local GPU) -> scripted recording (Playwright, production)
# -> captions, badges and mix (ffmpeg). Output: /data/users3/nshaik3/Projects/Oculomics/RetiLink/demo/InsightRx_demo.mp4
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/data/users3/nshaik3/Projects/Oculomics/RetiLink/venv/bin/python
export HF_HOME=/data/users3/nshaik3/.cache/huggingface HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} CUDA_VISIBLE_DEVICES=${GPU:-1}
$PY scripts/demo/narrate.py
$PY scripts/demo/record.py "$@"
$PY scripts/demo/compose.py
