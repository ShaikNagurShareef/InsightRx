"""
Build a self-contained, CPU-ready zip of Insight Rx for another computer (no GPU or cluster needed).

Includes: every tracked project file (code, docs, demo video, PyTorch weights), test_data/ sample photos, and the ONNX
model bundle as models/insightrx-onnx-v1. Excludes: the mBRSET/BRSET datasets, .git, virtual environments, caches,
local databases and secrets (.env).

  python scripts/make_portable_zip.py [--bundle DIR] [--out FILE]
"""
import argparse
import hashlib
import os
import subprocess
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUNDLE = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/export/insightrx-onnx-v1"
DEFAULT_OUT = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/package/InsightRx-portable.zip"
TOP = "InsightRx"
STORE = (".onnx", ".pth", ".joblib", ".mp4", ".jpg", ".jpeg", ".png", ".woff2", ".zip", ".pdf")
NEVER = (".env", ".vercel/", ".claude/", ".agents/", "__pycache__/", ".pytest_cache/")


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return [f for f in out.split("\0") if f and not f.startswith(NEVER) and os.path.isfile(os.path.join(ROOT, f))]


def walk(base):
    for d, _, files in os.walk(base):
        for f in files:
            if "__pycache__" not in d:
                yield os.path.join(d, f)


def is_lfs_pointer(path):
    if os.path.getsize(path) > 1024:
        return False
    with open(path, "rb") as fh:
        return fh.read(40).startswith(b"version https://git-lfs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    entries = [(os.path.join(ROOT, f), f) for f in tracked_files()]
    entries += [(p, os.path.relpath(p, ROOT)) for p in walk(os.path.join(ROOT, "test_data"))]
    entries += [(p, os.path.join("models", os.path.basename(args.bundle.rstrip("/")), os.path.relpath(p, args.bundle)))
                for p in walk(args.bundle)]
    entries = sorted({arc: src for src, arc in entries}.items())
    pointers = [arc for arc, src in entries if is_lfs_pointer(src)]
    if pointers:
        raise SystemExit(f"Git LFS pointers found instead of real files (run 'git lfs pull'): {pointers[:5]}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".part"
    total = sum(os.path.getsize(src) for _, src in entries)
    print(f"{len(entries)} files, {total / 2**30:.2f} GB uncompressed -> {args.out}")
    t = time.time()
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for arc, src in entries:
            kind = zipfile.ZIP_STORED if arc.lower().endswith(STORE) else zipfile.ZIP_DEFLATED
            info = zipfile.ZipInfo.from_file(src, f"{TOP}/{arc}")
            info.compress_type = kind
            if arc.endswith((".sh",)):
                info.external_attr = 0o755 << 16          # keep shell scripts executable after unzip
            with open(src, "rb") as fh, z.open(info, "w", force_zip64=True) as out:
                while chunk := fh.read(8 << 20):
                    out.write(chunk)
    os.replace(tmp, args.out)
    h = hashlib.sha256()
    with open(args.out, "rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
    with open(args.out + ".sha256", "w") as fh:
        fh.write(f"{h.hexdigest()}  {os.path.basename(args.out)}\n")
    print(f"wrote {os.path.getsize(args.out) / 2**30:.2f} GB in {time.time() - t:.0f} s; sha256 {h.hexdigest()[:16]}…")


if __name__ == "__main__":
    main()
