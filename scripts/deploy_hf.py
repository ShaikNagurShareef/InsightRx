"""
Deploy Insight Rx inference to Hugging Face (an alternative to the GPU machine + Cloudflare tunnel).
Note: running a Docker Space on a personal account requires Hugging Face PRO (free accounts get HTTP 402);
the model upload (step 1) works on any account. Use --skip-space to upload the models only.

  1. uploads the ONNX bundle (scripts/export_models.py) to a PRIVATE model repo   <user>/insightrx-onnx
  2. creates / updates a PRIVATE Docker Space                                   <user>/insightrx-vision
     (free CPU basic; downloads the bundle at startup; serves insightrx.vision_api)
  3. sets the Space secrets HF_TOKEN (read the model repo) and INSIGHTRX_VISION_KEY (shared with the web app),
     and the variables INSIGHTRX_MODEL_REPO, INSIGHTRX_ORT_THREADS=2, INSIGHTRX_WARMUP=1

Point the web app at it with INSIGHTRX_VISION_URL=https://<user>-insightrx-vision.hf.space, INSIGHTRX_VISION_ASYNC=1 and
INSIGHTRX_VISION_HF_TOKEN (a token that can read the private Space).

  python scripts/deploy_hf.py [--bundle DIR] [--skip-models] [--code-only]
Uses the Hugging Face token of this machine (huggingface-cli login); INSIGHTRX_VISION_KEY from the environment or .env.
"""
import argparse
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi, get_token

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUNDLE = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/export/insightrx-onnx-v1"
SPACE_CODE = ["insightrx/__init__.py", "insightrx/vision_api.py", "insightrx/app/__init__.py", "insightrx/app/vision.py",
              "insightrx/app/remote_vision.py", "insightrx/ml/__init__.py", "insightrx/ml/preprocess.py"]
MODEL_CARD = """---
license: other
tags: [onnx, ophthalmology, diabetic-retinopathy, oculomics]
---
# Insight Rx ONNX bundle (private)

CPU-portable inference bundle for Insight Rx (HackGT 13, team Coding Claws): DINOv2-L + LoRA retina ensemble
(2 seeds, LoRA merged), frozen systemic encoders, scikit-learn systemic heads, calibration and a manifest with checksums.
Exported by `scripts/export_models.py`; parity with PyTorch is recorded in `PARITY.json`.

Private: trained on mBRSET (PhysioNet credentialed data); the RETFound encoders are under their authors' non-commercial
terms. Research prototype; outputs support clinician review and are not diagnoses.
"""


def env_value(name):
    if os.environ.get(name):
        return os.environ[name]
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p):
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--skip-models", action="store_true", help="the model repo is already up to date")
    ap.add_argument("--code-only", action="store_true", help="only refresh the Space code")
    ap.add_argument("--skip-space", action="store_true", help="upload the models only (no Space)")
    args = ap.parse_args()
    api, token = HfApi(), get_token()
    if not token:
        sys.exit("No Hugging Face token on this machine: run `huggingface-cli login` first.")
    user = api.whoami()["name"]
    model_repo, space = f"{user}/insightrx-onnx", f"{user}/insightrx-vision"
    key = env_value("INSIGHTRX_VISION_KEY")
    if not key:
        sys.exit("INSIGHTRX_VISION_KEY is not set (environment or .env).")

    if not (args.skip_models or args.code_only):
        api.create_repo(model_repo, repo_type="model", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md", repo_id=model_repo)
        print(f"uploading {args.bundle} -> {model_repo} (resumable)")
        api.upload_large_folder(repo_id=model_repo, folder_path=args.bundle, repo_type="model")

    if args.skip_space:
        print(f"models: https://huggingface.co/{model_repo}")
        return
    api.create_repo(space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as stage:
        for f in os.listdir(os.path.join(ROOT, "deploy", "hf_space")):
            shutil.copy(os.path.join(ROOT, "deploy", "hf_space", f), stage)
        for f in SPACE_CODE:
            os.makedirs(os.path.join(stage, os.path.dirname(f)), exist_ok=True)
            shutil.copy(os.path.join(ROOT, f), os.path.join(stage, f))
        api.upload_folder(folder_path=stage, repo_id=space, repo_type="space", commit_message="Deploy Insight Rx vision worker")
    if not args.code_only:
        api.add_space_secret(space, "HF_TOKEN", token)
        api.add_space_secret(space, "INSIGHTRX_VISION_KEY", key)
        for k, v in {"INSIGHTRX_MODEL_REPO": model_repo, "INSIGHTRX_ORT_THREADS": "2", "INSIGHTRX_WARMUP": "1"}.items():
            api.add_space_variable(space, k, v)
    host = space.replace("/", "-").replace("_", "-").lower()
    print(f"Space: https://huggingface.co/spaces/{space}")
    print(f"API:   https://{host}.hf.space   (set INSIGHTRX_VISION_URL to this)")


if __name__ == "__main__":
    main()
