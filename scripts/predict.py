"""
Run the Insight Rx models on any folder of fundus photographs (new images, external datasets).

  python scripts/predict.py IMAGES_DIR [--out results.csv] [--maps MAPS_DIR] [--labels labels.csv]

  IMAGES_DIR   folder with .jpg/.jpeg/.png (searched recursively)
  --maps DIR   also save a DR attention overlay per image (PNG)
  --labels CSV optional ground truth: columns `image` (file name) and `dr_referable` (1 = ICDR >= 2), or `icdr` (0-4).
               If given, prints AUROC / sensitivity / specificity at the frozen mBRSET threshold, so you can measure
               how well the model transfers to the new dataset (e.g. APTOS 2019, Messidor-2, IDRiD, EyePACS).

Uses the same weights, calibration and thresholds as the app (./weights, or INSIGHTRX_MODEL_DIR).
Scores are per photo; the app's two-eye rule needs laterality, so use a case for a full assessment.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from insightrx.app.vision import VisionService  # noqa: E402
from insightrx.app.vision import MODEL_DIR  # noqa: E402

ICDR = ["No apparent DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images")
    ap.add_argument("--out", default="insightrx_predictions.csv")
    ap.add_argument("--maps", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.images) for f in fs
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not paths:
        sys.exit(f"no images found in {args.images}")
    svc = VisionService(MODEL_DIR)
    if svc.mode != "live":
        sys.exit(f"no trained model found in {MODEL_DIR}")
    print(f"{len(paths)} images, model {svc.version}, device {svc.device}")
    if args.maps:
        os.makedirs(args.maps, exist_ok=True)

    rows, t0 = [], time.time()
    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        res = svc.analyze([{"id": j, "path": p, "laterality": "unknown"} for j, p in enumerate(chunk)], {})
        for j, p in enumerate(chunk):
            r = res["images"][j]
            row = {"image": os.path.relpath(p, args.images), "suitable": r.get("quality") != "unsupported",
                   "quality": r.get("quality"), "p_unusable": r.get("p_quality_poor"),
                   "dr_score": r.get("p_dr") if r.get("quality") != "unassessable" else None,
                   "dr_score_ungated": r.get("p_dr"), "dr_referable_flag": r.get("dr_positive"),
                   "icdr_estimate": r.get("icdr_grade"), "icdr_name": ICDR[r["icdr_grade"]] if "icdr_grade" in r else "",
                   "edema_score": r.get("p_edema"), "reasons": "; ".join(r.get("reasons", []))}
            for g, pr in enumerate(r.get("icdr_probs", [])):
                row[f"p_icdr{g}"] = pr
            rows.append(row)
            if args.maps and r.get("quality") in ("assessable", "uncertain"):
                png = svc.explain(p, "dr_referable")
                if png:
                    with open(os.path.join(args.maps, os.path.splitext(os.path.basename(p))[0] + "_attention.png"), "wb") as fh:
                        fh.write(png)
        print(f"  {min(i + args.batch, len(paths))}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)

    cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    thr = svc.calib["thresholds"]
    print(f"frozen thresholds: unusable >= {thr['quality_poor']:.3f}, referable DR >= {thr['dr_referable_patient']:.3f}")

    if args.labels:
        import numpy as np
        from sklearn.metrics import roc_auc_score
        lab = {}
        with open(args.labels) as fh:
            for r in csv.DictReader(fh):
                key = os.path.basename(r["image"])
                if "dr_referable" in r and r["dr_referable"] != "":
                    lab[key] = float(r["dr_referable"])
                elif "icdr" in r and r["icdr"] != "":
                    lab[key] = float(float(r["icdr"]) >= 2)
        y, s, gated = [], [], 0
        for r in rows:
            k = os.path.basename(r["image"])
            if k not in lab:
                continue
            if r["dr_score"] is None or r["quality"] in ("unassessable", "unsupported"):   # gated, as in the app
                gated += 1
                continue
            y.append(lab[k])
            s.append(r["dr_score"])
        y, s = np.array(y), np.array(s)
        if len(np.unique(y)) < 2:
            sys.exit("labels need both classes to evaluate")
        pred = s >= thr["dr_referable_patient"]
        print(f"\nexternal evaluation on {len(y)} labelled photos ({int(y.sum())} referable; {gated} rejected by gates)")
        print(f"  AUROC        {roc_auc_score(y, s):.3f}")
        print(f"  sensitivity  {(pred & (y == 1)).sum() / max((y == 1).sum(), 1):.3f}")
        print(f"  specificity  {(~pred & (y == 0)).sum() / max((y == 0).sum(), 1):.3f}")
        print("  (threshold frozen on mBRSET validation; a different camera/population may need recalibration)")


if __name__ == "__main__":
    main()
