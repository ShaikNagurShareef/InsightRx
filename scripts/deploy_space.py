"""
Deploy RetiLink to a Hugging Face Docker Space (private by default).

  python scripts/deploy_space.py [--space <user>/RetiLink] [--public]

Uploads retilink/, scripts/, weights/ (model checkpoints + aggregate metrics only) and the Space Dockerfile.
No mBRSET images or per-patient outputs are uploaded; the Space seeds a synthetic workspace without images.
The weights were trained on credentialed PhysioNet data: keep the Space private unless redistribution of the
trained model has been cleared. Set GEMINI_API_KEY as a Space secret to enable LLM evidence briefs.
"""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_README = """---
title: RetiLink
emoji: 👁️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Retinal screening to accountable HCP-to-HCP consultation
---

# RetiLink

HackGT 13 · Impiricus challenge. Portable retinal screening → clinician review → signed specialist consultation →
tracked handoff. Research prototype: synthetic patients, simulated workflow data, model outputs estimate mBRSET dataset
labels for clinician review and are not diagnoses.

Sign in with any demo account, create a case and upload fundus photographs you are authorised to use.
Hardware: a T4 GPU keeps analysis to a few seconds; on free CPU each image takes about a minute.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=None)
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    space = args.space or f"{user}/RetiLink"
    api.create_repo(space, repo_type="space", space_sdk="docker", private=not args.public, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        ign = shutil.ignore_patterns("__pycache__", "*.pyc")
        for d in ("retilink", "scripts", "weights"):
            shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d), ignore=ign)
        os.makedirs(os.path.join(tmp, "deploy", "space"))
        shutil.copy(os.path.join(ROOT, "deploy", "space", "requirements.txt"), os.path.join(tmp, "deploy", "space"))
        shutil.copy(os.path.join(ROOT, "deploy", "space", "Dockerfile"), os.path.join(tmp, "Dockerfile"))
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(SPACE_README)
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space",
                          commit_message="Deploy RetiLink", delete_patterns=["retilink/**", "scripts/**"])
    print(f"https://huggingface.co/spaces/{space}  (private={not args.public})")


if __name__ == "__main__":
    main()
