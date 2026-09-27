"""
Export Insight Rx models into a self-contained, device-portable bundle (no Hugging Face downloads, no training code).

  bundle/
    manifest.json                  preprocessing, head layout, calibration, thresholds, versions, licences, checksums
    retina/seed42.onnx, seed43.onnx    DINOv2-L with LoRA merged into the weights + multi-task heads
                                       input  pixel_values [B, 3, 392, 392] float32 (ImageNet-normalised)
                                       output logits [B, 11] (heads in manifest order), embedding [B, 2048]
    encoders/<backbone>.onnx       frozen encoders for the systemic heads -> features [B, 2048]
    systemic/systemic_cv.joblib    scikit-learn heads (pin scikit-learn to the manifest version)
    calibration.json, metrics/     temperatures, frozen thresholds, measured metrics
    PARITY.json                    ONNX vs PyTorch agreement on real images

Runs anywhere ONNX Runtime does: CPU (x86/ARM), NVIDIA CUDA, Apple CoreML, Windows DirectML.
Load it with INSIGHTRX_MODEL_DIR=<bundle> (insightrx.app.vision picks the ONNX backend automatically).

  python scripts/export_models.py [--out DIR] [--no-encoders] [--check-images a.jpg b.jpg ...]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from insightrx.ml import config as C  # noqa: E402
from insightrx.ml.model import LoRALinear, RetiModel  # noqa: E402

WEIGHTS = os.path.join(ROOT, "weights")
DEFAULT_OUT = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/export/insightrx-onnx-v1"
OPSET = 17
LICENCES = {
    "retina": "Insight Rx fine-tune of facebook/dinov2-large (Apache-2.0). Trained on mBRSET (PhysioNet credentialed data).",
    "dinov2-large_224px": "facebook/dinov2-large, Apache-2.0.",
    "RETFound_dinov2_meh_224px": "RETFound (YukunZhou/RETFound_dinov2_meh), gated; authors' non-commercial terms. Do not redistribute publicly.",
    "RETFound_dinov2_meh_448px": "RETFound (YukunZhou/RETFound_dinov2_meh), gated; authors' non-commercial terms. Do not redistribute publicly.",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merged_linear(m):
    """One LoRALinear folded into a plain Linear: W' = W + scale * B @ A (exact, no runtime adapter)."""
    lin = nn.Linear(m.base.in_features, m.base.out_features, bias=m.base.bias is not None)
    with torch.no_grad():
        lin.weight.copy_(m.base.weight + m.scale * (m.B.weight @ m.A.weight))
        if m.base.bias is not None:
            lin.bias.copy_(m.base.bias)
    return lin


def merge_lora(model):
    """Fold every LoRA adapter of the backbone into its base Linear."""
    n = 0
    for blk in model.backbone.encoder.layer:
        att = blk.attention.attention
        for name in ("query", "value"):
            m = getattr(att, name)
            if isinstance(m, LoRALinear):
                setattr(att, name, merged_linear(m))
                n += 1
    return n


class RetinaGraph(nn.Module):
    """Export wrapper: concatenated head logits (manifest order) + the pooled embedding."""

    def __init__(self, model, heads):
        super().__init__()
        self.model, self.names = model, [h[0] for h in heads]

    def forward(self, pixel_values):
        logits, emb = self.model(pixel_values, return_embedding=True)
        return torch.cat([logits[n] for n in self.names], 1), emb


def load_seed(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = RetiModel(heads=ck["heads"], grad_checkpointing=False)
    m.load_state_dict(ck["state_dict"], strict=False)
    return m.eval(), ck


def export(module, dummy, path, inputs, outputs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(module, (dummy,), path, input_names=inputs, output_names=outputs, opset_version=OPSET,
                      dynamic_axes={inputs[0]: {0: "batch"}, **{o: {0: "batch"} for o in outputs}},
                      do_constant_folding=True, dynamo=False)
    import onnx
    onnx.checker.check_model(path)
    return path


def export_retina(out, calib):
    seeds = [s for s in calib["seeds"]]
    files, heads, size = [], None, None
    for seed in seeds:
        m, ck = load_seed(os.path.join(WEIGHTS, "ckpt", f"image_seed{seed}.pth"))
        merged = merge_lora(m)
        heads, size = ck["heads"], ck["image_size"]
        path = os.path.join(out, "retina", f"seed{seed}.onnx")
        t = time.time()
        export(RetinaGraph(m, heads).eval(), torch.randn(1, 3, size, size), path, ["pixel_values"], ["logits", "embedding"])
        print(f"retina seed {seed}: merged {merged} LoRA layers, exported in {time.time() - t:.0f} s "
              f"({os.path.getsize(path) / 2**20:.0f} MB)")
        files.append(path)
    return files, heads, size


def export_encoders(out, backbones):
    from insightrx.ml.frozen import BACKBONES, FrozenEncoder
    files = {}
    for bb in sorted(backbones):
        enc = FrozenEncoder(bb).eval()
        size = BACKBONES[bb][1]
        path = os.path.join(out, "encoders", f"{bb}.onnx")
        t = time.time()
        export(enc, torch.randn(1, 3, size, size), path, ["pixel_values"], ["features"])
        print(f"encoder {bb}: exported in {time.time() - t:.0f} s ({os.path.getsize(path) / 2**20:.0f} MB)")
        files[bb] = (path, size)
    return files


def parity(out, retina_files, enc_files, images):
    """Compare ONNX Runtime (CPU, fp32) with PyTorch (CPU, fp32) on real images through the shared preprocessing."""
    import onnxruntime as ort
    from insightrx.ml.preprocess import preprocess
    report = {"images": [os.path.basename(p) for p in images], "retina": {}, "encoders": {}}
    x = np.stack([preprocess(p, C.IMAGE_SIZE) for p in images])
    for path in retina_files:
        seed = int(os.path.basename(path)[4:-5])
        m, _ = load_seed(os.path.join(WEIGHTS, "ckpt", f"image_seed{seed}.pth"))
        heads = m.heads
        with torch.no_grad():
            lo, emb = RetinaGraph(m, heads)(torch.from_numpy(x))
        so = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ol, oe = so.run(None, {"pixel_values": x})
        report["retina"][f"seed{seed}"] = {"max_abs_logit_diff": float(np.abs(ol - lo.numpy()).max()),
                                           "max_abs_embedding_diff": float(np.abs(oe - emb.numpy()).max())}
        print(f"parity seed {seed}: logits {report['retina'][f'seed{seed}']['max_abs_logit_diff']:.2e}")
    from insightrx.ml.frozen import FrozenEncoder
    for bb, (path, size) in enc_files.items():
        xe = np.stack([preprocess(p, size, draft=224) for p in images])
        with torch.no_grad():
            ref = FrozenEncoder(bb).eval()(torch.from_numpy(xe)).numpy()
        got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"pixel_values": xe})[0]
        report["encoders"][bb] = {"max_abs_feature_diff": float(np.abs(got - ref).max())}
        print(f"parity {bb}: features {report['encoders'][bb]['max_abs_feature_diff']:.2e}")
    json.dump(report, open(os.path.join(out, "PARITY.json"), "w"), indent=1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-encoders", action="store_true", help="skip the systemic encoders (retina only)")
    ap.add_argument("--check-images", nargs="*", default=sorted(glob.glob(
        os.path.join(ROOT, "test_data", "mbrset", "patients", "03_referable_both_eyes", "*.jpg")))[:4])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    calib = json.load(open(os.path.join(WEIGHTS, "calibration.json")))
    retina_files, heads, size = export_retina(args.out, calib)

    import joblib
    import sklearn
    sys_src = os.path.join(WEIGHTS, "ckpt", "systemic_cv.joblib")
    bundle = joblib.load(sys_src)
    backbones = {h["backbone"] for h in bundle["heads"].values() if h["backbone"]}
    enc_files = {} if args.no_encoders else export_encoders(args.out, backbones)
    os.makedirs(os.path.join(args.out, "systemic"), exist_ok=True)
    shutil.copy(sys_src, os.path.join(args.out, "systemic", "systemic_cv.joblib"))
    shutil.copy(os.path.join(WEIGHTS, "calibration.json"), args.out)
    shutil.copytree(os.path.join(WEIGHTS, "metrics"), os.path.join(args.out, "metrics"), dirs_exist_ok=True)

    report = parity(args.out, retina_files, enc_files, args.check_images) if args.check_images else None
    files = sorted(os.path.relpath(p, args.out) for p in glob.glob(os.path.join(args.out, "**", "*"), recursive=True)
                   if os.path.isfile(p) and not p.endswith("manifest.json"))
    manifest = {
        "name": "insightrx-onnx", "format_version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "opset": OPSET,
        "retina": {"files": [os.path.relpath(p, args.out) for p in retina_files], "image_size": size,
                   "heads": [{"name": n, "type": t, "outputs": k} for n, t, k, _ in heads],
                   "test_time_augmentation": "mean of logits over the image and its horizontal flip, then across seeds",
                   "embedding_dim": 2048},
        "encoders": {bb: {"file": os.path.relpath(p, args.out), "image_size": s, "jpeg_draft": 224}
                     for bb, (p, s) in enc_files.items()},
        "preprocessing": {"decode": "PIL, JPEG draft to 2x image size, RGB", "crop": "FundusCrop(threshold=15, min_fraction=0.01), pad to square with black",
                          "resize": "PIL bicubic to image_size x image_size", "scale": "divide by 255",
                          "normalise": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "layout": "NCHW float32"},
        "calibration": "calibration.json (temperatures, frozen thresholds, seeds)",
        "systemic": {"file": "systemic/systemic_cv.joblib", "scikit_learn": sklearn.__version__,
                     "metadata_features": bundle["metadata_features"]},
        "runtime_requirements": ["onnxruntime>=1.17 (or onnxruntime-gpu / onnxruntime-silicon / onnxruntime-directml)",
                                 "numpy", "pillow", f"scikit-learn=={sklearn.__version__}", "joblib"],
        "licences": {k: v for k, v in LICENCES.items() if k == "retina" or k in enc_files},
        "parity": report,
        "sha256": {f: sha256(os.path.join(args.out, f)) for f in files},
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in files) / 2**30
    print(f"bundle: {args.out} ({len(files)} files, {total:.2f} GB)")


if __name__ == "__main__":
    main()
