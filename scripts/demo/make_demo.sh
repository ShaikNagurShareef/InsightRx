#!/bin/bash
# Regenerate the Insight Rx demo video: narration (VibeVoice, local GPU, Whisper-aligned) -> scripted recording (Playwright, production)
# -> captions, badges and mix (ffmpeg). Output: /data/users3/nshaik3/Projects/Oculomics/RetiLink/demo/InsightRx_demo.mp4
# One-time voice setup (separate venv; VibeVoice pins transformers 4.51.3):
#   git clone https://github.com/vibevoice-community/VibeVoice tools/VibeVoice
#   python -m venv vv-venv && vv-venv/bin/pip install torch==2.6.0 torchaudio soundfile faster-whisper -e tools/VibeVoice
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/data/users3/nshaik3/Projects/Oculomics/RetiLink/venv/bin/python
export HF_HOME=/data/users3/nshaik3/.cache/huggingface HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} CUDA_VISIBLE_DEVICES=${GPU:-1}
TTS=/data/users3/nshaik3/Projects/Oculomics/RetiLink/vv-venv/bin/python      # VibeVoice + faster-whisper alignment/QA
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=${TTS_GPUS:-0,1} ASR_GPU_INDEX=${ASR_GPU_INDEX:-1} $TTS scripts/demo/narrate.py
$PY scripts/demo/record.py "$@"
$PY scripts/demo/compose.py
