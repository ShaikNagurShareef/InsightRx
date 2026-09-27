"""Portable inference: torch-free preprocessing and head maths match training code; LoRA merge is exact;
an exported ONNX bundle (if INSIGHTRX_ONNX_BUNDLE points at one) loads and verifies its checksums."""
import glob
import hashlib
import json
import os
import sys

import tempfile

import numpy as np
import pytest
import torch

# same isolation as test_workflow: the app's default service must stay SIMULATED during the suite
os.environ.setdefault("INSIGHTRX_MODEL_DIR", os.path.join(tempfile.mkdtemp(prefix="insightrx_test_"), "no_model"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from insightrx.app.vision import logit, ordinal_probs, split_heads  # noqa: E402
from insightrx.ml import config as C  # noqa: E402
from insightrx.ml.model import LoRALinear, ordinal_probs_np, split_logits  # noqa: E402
from insightrx.ml.preprocess import preprocess  # noqa: E402

SAMPLES = sorted(glob.glob(os.path.join(ROOT, "test_data", "mbrset", "patients", "*", "*.jpg")))[:2]


def _fundus(tmp_path):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (900, 700), (0, 0, 0))
    ImageDraw.Draw(im).ellipse((120, 20, 780, 680), fill=(190, 90, 40))
    p = tmp_path / "f.jpg"
    im.save(p, quality=92)
    return str(p)


@pytest.mark.parametrize("size,draft", [(392, None), (448, 224), (224, 224)])
def test_preprocess_matches_torchvision_eval_transform(tmp_path, size, draft):
    from insightrx.ml.data import eval_transform, open_fundus
    for p in SAMPLES or [_fundus(tmp_path)]:
        ref = eval_transform(size)(open_fundus(p, draft or size)).numpy()
        assert np.array_equal(preprocess(p, size, draft=draft), ref)


def test_numpy_head_maths_match_training_code():
    z = np.random.RandomState(0).randn(6, sum(k for _, _, k, _ in C.HEADS))
    ref = split_logits(z)
    got = split_heads(z, C.HEADS)
    assert all(np.array_equal(ref[n], got[n]) for n in ref)
    assert np.allclose(ordinal_probs(z[:, :4]), ordinal_probs_np(z[:, :4]))
    s = np.array([0.0, 0.2, 0.5, 1.0])
    assert np.allclose(logit(s)[:, 0], np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))))


def test_lora_merge_is_exact():
    from export_models import merged_linear
    torch.manual_seed(0)
    m = LoRALinear(torch.nn.Linear(64, 32), rank=8).eval()
    torch.nn.init.normal_(m.B.weight)                   # non-zero adapter
    x = torch.randn(5, 64)
    assert torch.allclose(m(x), merged_linear(m)(x), atol=1e-5)


BUNDLE = os.environ.get("INSIGHTRX_ONNX_BUNDLE")


@pytest.mark.skipif(not BUNDLE, reason="set INSIGHTRX_ONNX_BUNDLE to an exported bundle to run")
def test_exported_bundle_loads_and_verifies():
    man = json.load(open(os.path.join(BUNDLE, "manifest.json")))
    for f, digest in man["sha256"].items():
        if f.endswith((".json", ".joblib")):              # small files: full check (ONNX checked by size to stay fast)
            assert hashlib.sha256(open(os.path.join(BUNDLE, f), "rb").read()).hexdigest() == digest
        assert os.path.getsize(os.path.join(BUNDLE, f)) > 0
    assert max(v["max_abs_logit_diff"] for v in man["parity"]["retina"].values()) < 1e-3
    from insightrx.app.vision import VisionService
    svc = VisionService(model_dir=BUNDLE)
    assert svc.backend == "onnx" and svc.mode == "live" and len(svc.models) == len(man["retina"]["files"])
