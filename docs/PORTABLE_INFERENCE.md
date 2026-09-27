# Running Insight Rx inference on another device

The trained models can be exported into a **self-contained ONNX bundle** that runs without PyTorch, transformers, a GPU, or any Hugging Face download. The same bundle runs on CPU (x86 or ARM), NVIDIA GPUs, Apple Silicon and Windows GPUs through ONNX Runtime. It reproduces the full pipeline: quality gate, referable DR, ICDR grade, macular edema, eye and patient aggregation, and all 9 systemic heads.

## 1. Export (on the training machine)

```bash
python scripts/export_models.py            # -> /data/users3/nshaik3/Projects/Oculomics/RetiLink/export/insightrx-onnx-v1
python scripts/export_models.py --out ./insightrx-onnx-v1 --no-encoders   # retina only (2.4 GB, no RETFound)
```

The exporter:
- **folds the LoRA adapters into the DINOv2 weights,** 48 layers per seed, exactly, so no adapter code is needed at runtime;
- exports each seed and each frozen systemic encoder to ONNX (opset 17, dynamic batch size);
- writes a `manifest.json` with the preprocessing, head layout, calibration, versions, licences and SHA-256 checksums;
- runs a PyTorch-vs-ONNX **parity check** on real patient images and records it in `PARITY.json`.

| File | Size | What it is |
|---|---|---|
| `retina/seed42.onnx`, `retina/seed43.onnx` | 1.2 GB each | DINOv2-L with LoRA merged, plus multi-task heads. Input: `pixel_values [B,3,392,392]`. Outputs: `logits [B,11]` and `embedding [B,2048]`. |
| `encoders/dinov2-large_224px.onnx` | 1.2 GB | Frozen DINOv2 encoder for one systemic head (Apache-2.0) |
| `encoders/RETFound_dinov2_meh_224px.onnx`, `_448px.onnx` | 1.2 GB each | Frozen RETFound encoders for three systemic heads (gated, non-commercial terms, **do not publish**) |
| `systemic/systemic_cv.joblib` | 4 MB | scikit-learn systemic heads (pin scikit-learn to the manifest version) |
| `calibration.json`, `metrics/` | small | Temperatures, frozen thresholds and measured metrics |
| `manifest.json`, `PARITY.json` | small | Contract and verification |

Total: **5.7 GB** (2.4 GB for the retina-only bundle). Heads whose encoder is missing from a bundle are reported as *Not evaluated*.

## 2. Verified parity

On 4 real mBRSET photos, running on CPU in fp32:

| Component | Max difference ONNX vs PyTorch |
|---|---|
| Retina seed 42 / 43 logits | 7.6e-06 / 1.3e-05 |
| Encoder features (RETFound 224 / 448, DINOv2 224) | 2.8e-05 / 1.7e-05 / 5.1e-05 |
| End to end, one patient: per-image probabilities | 4.8e-07 |
| End to end: verdicts, grades, edema flags and all 9 systemic statuses | **identical** |

The preprocessing (`insightrx/ml/preprocess.py`, NumPy and Pillow only) is **bit-identical** to the training `eval_transform`.

## 3. Run on the other device

Copy the bundle (for example `rsync -a insightrx-onnx-v1/ user@device:~/insightrx-onnx-v1/`) and this repository. Then:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-runtime.txt          # onnxruntime, numpy, pillow, scikit-learn, joblib, pandas; no torch

# score a folder of fundus photos -> CSV
INSIGHTRX_MODEL_DIR=~/insightrx-onnx-v1 python scripts/predict.py path/to/photos --out predictions.csv
```

From Python:

```python
from insightrx.app.vision import VisionService
svc = VisionService(model_dir="~/insightrx-onnx-v1")         # picks the ONNX backend from manifest.json
result = svc.analyze([{"id": 1, "path": "OD.jpg", "laterality": "OD"},
                      {"id": 2, "path": "OS.jpg", "laterality": "OS"}],
                     {"age": 64, "sex": 1, "dm_time": 12, "insulin": 1, "oraltreatment_dm": 1})
print(result["overall"], result["eyes"], result["systemic"]["systemic_hypertension"])
```

To serve the live web app from that device instead of the GPU workstation, add `pip install fastapi uvicorn python-multipart` and run:

```bash
INSIGHTRX_MODEL_DIR=~/insightrx-onnx-v1 bash scripts/run_vision_tunnel.sh
```

### Choosing the hardware

| Device | Install | Provider picked automatically |
|---|---|---|
| Any CPU (Linux, Windows, macOS, ARM) | `onnxruntime` | `CPUExecutionProvider` |
| NVIDIA GPU (CUDA 12) | `onnxruntime-gpu` | `CUDAExecutionProvider` |
| Apple Silicon | `onnxruntime` | `CoreMLExecutionProvider`, with CPU fallback |
| Windows GPU (AMD, Intel, NVIDIA) | `onnxruntime-directml` | `DmlExecutionProvider` |

To force a provider, set `INSIGHTRX_ORT_PROVIDERS=CPUExecutionProvider`. On a 12-core CPU, one patient with 4 photos takes about 60 s including test-time augmentation and both seeds; a GPU is much faster.

## 4. What differs from the PyTorch backend

- **Attention maps** ("Where the model looked") need gradients, so they are only available with the PyTorch weights. Every prediction, aggregation and systemic score is identical.
- The live GSU worker keeps the PyTorch backend, which runs in bf16 on the GPU.

## 5. Licences and sharing

- The retina models are a fine-tune of DINOv2 (Apache-2.0), trained on **mBRSET**, which is credentialed PhysioNet data. Share them according to your data-use obligations.
- The **RETFound encoders** come from a gated model under non-commercial terms. Keep bundles that contain them private, or export with `--no-encoders` for anything public.
- The bundle is **not** stored in this Git repository. It is 5.7 GB, above GitHub's free LFS quota, and the RETFound terms would forbid it. Use `scripts/export_models.py` to regenerate it, or share it privately (for example a private Hugging Face model repo or `rsync`).

## 6. Hosted copy and remote workers

- **Model repo:** the bundle is published to the **private** Hugging Face repo `nagur-shareef-shaik/insightrx-onnx`. Any worker can fetch it with `huggingface_hub.snapshot_download(..., token=...)` instead of copying files by hand.
- **Live app worker:** the live app's vision worker runs on the GSU GPU workstation behind a Cloudflare tunnel (`scripts/run_vision_tunnel.sh`). It answers synchronously in seconds.
- **Slow workers:** for CPU-only hosts, the worker also exposes a job queue (`POST /jobs`, `GET /jobs/{id}`). Setting `INSIGHTRX_VISION_ASYNC=1` in the web app makes Screen and case analyses queue and poll:
  - pages show "Analysing…" and refresh themselves;
  - saving a screened patient reuses its result;
  - lost jobs are resubmitted.
- **Hugging Face Space:** `scripts/deploy_hf.py` and `deploy/hf_space/` deploy that worker as a private Docker Space that downloads the bundle at startup. Since 2026, Hugging Face requires **PRO** to run Docker or Gradio Spaces on a personal account, so that step returns 402 on a free account; the model upload works on any account.
