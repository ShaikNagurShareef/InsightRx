"""
Copy a labelled set of demo/test images from mBRSET (held-out test split) and BRSET into test_data/.

  python scripts/make_test_data.py

test_data/
  mbrset/singles/      2 photos per category: ICDR 0-4, macular edema, unusable quality (never seen in training)
  mbrset/patients/     6 complete patients (both eyes, 2 views each) + patient.json to fill in "New screening"
  brset/               other cameras (Canon CR, Nikon NF5050): DR grades, edema, hypertensive retinopathy,
                       inadequate quality, AMD, vascular occlusion - for domain-shift testing
  */manifest.csv       labels per image; usable directly with `scripts/predict.py --labels`

Both datasets are credentialed PhysioNet data: test_data/ is git-ignored and excluded from deployments.
"""
import json
import os
import random
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from insightrx.ml.data import load_tables  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data")
SPLIT = "/data/users3/nshaik3/Projects/Oculomics/OculoMoE/cache/mbrset/splits_protocol_P1.json"
BRSET = "/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/brazilian-ophthalmological/1.0.1/"
CONDS = ["systemic_hypertension", "nephropathy", "vascular_disease", "acute_myocardial_infarction", "neuropathy",
         "diabetic_foot", "obesity"]
rng = random.Random(5)


def mbrset():
    images, patients = load_tables()
    test = set(json.load(open(SPLIT))["splits"]["0"]["test"])
    t = images[images["patient_id"].isin(test)].copy()
    pat = patients.set_index("patient_id")
    out = os.path.join(ROOT, "mbrset", "singles")
    os.makedirs(out, exist_ok=True)
    rows = []
    good = t[t["quality_poor"] == 0]
    picks = [(f"icdr{g}", good[good["icdr"] == g]) for g in range(5)]
    picks += [("edema", good[good["edema"] == 1]), ("unusable", t[t["quality_poor"] == 1])]
    for cat, df in picks:
        for i, (_, r) in enumerate(df.sample(min(2, len(df)), random_state=rng.randint(0, 10 ** 6)).iterrows()):
            name = f"mbrset_{cat}_{i + 1}_{r['eye']}.jpg"
            shutil.copy(r["path"], os.path.join(out, name))
            rows.append({"image": name, "category": cat, "eye": r["eye"], "icdr": r["icdr"], "dr_referable": r["dr_referable"],
                         "edema": r["edema"], "quality_poor": r["quality_poor"], "source_file": r["image_id"]})
    pd.DataFrame(rows).to_csv(os.path.join(out, "manifest.csv"), index=False)

    # complete patients for "New screening" (both eyes); scenarios chosen from the labels
    g = {pid: d for pid, d in t.groupby("patient_id")}
    ok = {p: d for p, d in g.items() if (d["quality_poor"] == 0).all() and d["icdr"].notna().all()}
    by_eye = lambda d: d.groupby("eye")["icdr"].max()
    scen = {
        "01_no_dr": [p for p, d in ok.items() if (d["icdr"] == 0).all()],
        "02_mild_npdr": [p for p, d in ok.items() if d["icdr"].max() == 1],
        "03_referable_both_eyes": [p for p, d in ok.items() if (by_eye(d) >= 2).all()],
        "04_referable_one_eye": [p for p, d in ok.items() if sorted(by_eye(d) >= 2) == [False, True]],
        "05_unusable_quality": [p for p, d in g.items() if (d["quality_poor"] == 1).sum() >= 2],
        "06_systemic_rich": [p for p, d in ok.items() if pat.loc[p, "systemic_hypertension"] == 1 and
                             (pat.loc[p, ["nephropathy", "neuropathy", "diabetic_foot"]] == 1).sum() >= 2],
    }
    prow = []
    for name, cands in scen.items():
        if not cands:
            print("no patient for", name)
            continue
        pid = rng.choice(sorted(cands))
        d = g[pid].sort_values("image_id")
        folder = os.path.join(ROOT, "mbrset", "patients", name)
        os.makedirs(folder, exist_ok=True)
        counter = {"OD": 0, "OS": 0}
        for _, r in d.iterrows():
            counter[r["eye"]] += 1
            shutil.copy(r["path"], os.path.join(folder, f"{r['eye']}_{counter[r['eye']]}.jpg"))
            prow.append({"image": f"{name}/{r['eye']}_{counter[r['eye']]}.jpg", "scenario": name, "eye": r["eye"],
                         "icdr": r["icdr"], "dr_referable": r["dr_referable"], "edema": r["edema"],
                         "quality_poor": r["quality_poor"]})
        rec = pat.loc[pid]
        val = lambda v: "unknown" if v != v else ("present" if v == 1 else "absent")
        info = {"scenario": name, "age": None if rec["age"] != rec["age"] else int(rec["age"]),
                "sex": {1.0: "male", 0.0: "female"}.get(rec["sex"]),
                "years_with_diabetes": None if rec["dm_time"] != rec["dm_time"] else float(rec["dm_time"]),
                "insulin": val(rec["insulin"]).replace("present", "yes").replace("absent", "no"),
                "oral_diabetes_medication": val(rec["oraltreatment_dm"]).replace("present", "yes").replace("absent", "no"),
                "reported_history": {c: val(rec[c]) for c in CONDS},
                "ground_truth": {"max_icdr_per_eye": {e: None if v != v else int(v) for e, v in by_eye(d).items()},
                                 "unusable_photos": int(d["quality_poor"].sum())}}
        json.dump(info, open(os.path.join(folder, "patient.json"), "w"), indent=2)
    pd.DataFrame(prow).to_csv(os.path.join(ROOT, "mbrset", "patients", "manifest.csv"), index=False)
    return len(rows), len(prow)


def brset():
    df = pd.read_csv(os.path.join(BRSET, "labels_brset.csv"))
    out = os.path.join(ROOT, "brset")
    os.makedirs(out, exist_ok=True)
    adequate = df[df["quality"] == "Adequate"]
    picks = [(f"icdr{gr}", adequate[adequate["DR_ICDR"] == gr]) for gr in range(5)]
    picks += [("macular_edema", adequate[adequate["macular_edema"] == 1]),
              ("hypertensive_retinopathy", adequate[adequate["hypertensive_retinopathy"] == 1]),
              ("amd", adequate[adequate["amd"] == 1]), ("vascular_occlusion", adequate[adequate["vascular_occlusion"] == 1]),
              ("inadequate_quality", df[df["quality"] == "Inadequate"])]
    rows = []
    for cat, d in picks:
        for cam in ("Canon CR", "NIKON NF5050"):
            dd = d[d["camera"] == cam]
            if not len(dd):
                continue
            r = dd.sample(1, random_state=rng.randint(0, 10 ** 6)).iloc[0]
            src = os.path.join(BRSET, "fundus_photos", f"{r['image_id']}.jpg")
            if not os.path.exists(src):
                continue
            name = f"brset_{cat}_{'canon' if cam.startswith('Canon') else 'nikon'}.jpg"
            shutil.copy(src, os.path.join(out, name))
            rows.append({"image": name, "category": cat, "camera": cam, "eye": {1: "OD", 2: "OS"}.get(r["exam_eye"]),
                         "icdr": r["DR_ICDR"], "dr_referable": int(r["DR_ICDR"] >= 2), "macular_edema": r["macular_edema"],
                         "hypertensive_retinopathy": r["hypertensive_retinopathy"], "quality": r["quality"],
                         "source_file": r["image_id"]})
    pd.DataFrame(rows).to_csv(os.path.join(out, "manifest.csv"), index=False)
    return len(rows)


if __name__ == "__main__":
    s, p = mbrset()
    b = brset()
    print(f"test_data/: {s} mBRSET singles, {p} mBRSET patient photos, {b} BRSET photos")
