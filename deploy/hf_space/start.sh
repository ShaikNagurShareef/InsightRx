#!/usr/bin/env bash
# Fetch the ONNX bundle from the private model repo, then serve the vision worker on the Space port.
set -euo pipefail
python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
t = time.time()
snapshot_download(os.environ["INSIGHTRX_MODEL_REPO"], token=os.environ.get("HF_TOKEN"), local_dir="/home/user/models")
print(f"model bundle ready in {time.time() - t:.0f} s", flush=True)
PY
export INSIGHTRX_MODEL_DIR=/home/user/models
exec uvicorn insightrx.vision_api:app --host 0.0.0.0 --port 7860 --workers 1
