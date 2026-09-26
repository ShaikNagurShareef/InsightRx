"""
Systemic association models: 5-fold patient cross-validation over all 1,291 mBRSET patients.

Why CV instead of the P1 split: the test split holds only 5-47 positives for most conditions, and the fine-tuned
image model has seen the training patients' systemic labels. Here image features come from *frozen* foundation
encoders (no label exposure), so every patient is scored out-of-fold exactly once.

Per target we compare (all out-of-fold):
  metadata            age, sex, diabetes duration, insulin, oral treatment  -> logistic regression
  image:<backbone>    mean over quality-ok images of frozen [CLS, mean-patch] -> PCA -> logistic regression
  fused:<backbone>    both                                                  -> logistic regression
for backbones RETFound (224 / 448 px) and DINOv2-L (224 px). The chosen variant has the best OOF AUROC (selection
among 7 variants on the same folds is mildly optimistic - disclosed). Retinal added value (SG2) is the paired
patient-bootstrap AUROC difference chosen-image-containing variant minus metadata-only.

Explainability: grouped permutation importance on held-out folds (retinal image block vs each metadata field),
reliability curve of Platt-calibrated OOF scores, and occlusion baselines saved for per-patient explanations.

  python -m insightrx.ml.systemic_cv
Outputs: ckpt/systemic_cv.joblib, metrics/systemic_cv.json, metrics/systemic_cv_oof.csv
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import load_tables
from .frozen import BACKBONES, load_cached

TARGETS = C.SYSTEMIC_TARGETS + ["cardiovascular", "dm_complications"]
COMPOSITES = {"cardiovascular": ["vascular_disease", "acute_myocardial_infarction"],
              "dm_complications": ["nephropathy", "neuropathy", "diabetic_foot"]}
META = C.METADATA_FEATURES
N_PCA = 64
N_BOOT = 1000


def composite(p, cols):
    v = p[cols].values.astype(float)
    pos = np.nansum(v == 1, 1) > 0
    return np.where(pos, 1.0, np.where(~np.isnan(v).any(1), 0.0, np.nan))


def patient_features(images, patients):
    """Patient mean of frozen features over quality-ok images (all images if none is ok)."""
    out = {}
    for bb in BACKBONES:
        ids, F = load_cached(bb)
        df = pd.DataFrame(F, index=ids)
        im = images.set_index("image_id").loc[ids]
        ok = (im["quality_poor"] != 1).values
        good = df[ok].groupby(im["patient_id"].values[ok]).mean()
        allm = df.groupby(im["patient_id"].values).mean()
        out[bb] = good.reindex(patients["patient_id"]).fillna(allm.reindex(patients["patient_id"])).values.astype(np.float32)
    return out


def build(variant, n_img):
    if variant == "metadata":
        # regularised LR: the first run used gradient boosting, which overfit rare targets (nephropathy OOF AUROC
        # 0.35, below chance) and inflated the "retina adds value" contrast; LR is the robust baseline
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
    if variant.startswith("image:"):
        return make_pipeline(StandardScaler(), PCA(N_PCA, random_state=C.SEED),
                             LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
    ct = ColumnTransformer([
        ("img", make_pipeline(StandardScaler(), PCA(N_PCA, random_state=C.SEED)), slice(0, n_img)),
        ("meta", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
         slice(n_img, n_img + len(META)))])
    return make_pipeline(ct, LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))


def matrix(variant, feats, meta):
    if variant == "metadata":
        return meta
    bb = variant.split(":", 1)[1]
    return feats[bb] if variant.startswith("image:") else np.concatenate([feats[bb], meta], 1)


def boot(y, s, fn, rng):
    vals = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(fn(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def groups_for(variant, n_img):
    """Column groups for grouped permutation importance."""
    g = {}
    if variant != "metadata":
        g["retinal image"] = list(range(n_img))
    off = 0 if variant == "metadata" else n_img
    if not variant.startswith("image:"):
        for j, f in enumerate(META):
            g[f] = [off + j]
    return g


def platt(scores, y):
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1))[:, None], y)
    return lr


def logit(s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))[:, None]


def main():
    C.ensure_dirs()
    images, patients = load_tables()
    for t, cols in COMPOSITES.items():
        patients[t] = composite(patients, cols)
    feats = patient_features(images, patients)
    meta = patients[META].astype(float).values
    n_img = next(iter(feats.values())).shape[1]
    variants = ["metadata"] + [f"{k}:{bb}" for bb in BACKBONES for k in ("image", "fused")]
    rng = np.random.RandomState(C.SEED)

    report, bundle, oof_rows = {}, {"backbones": list(BACKBONES), "metadata_features": META, "heads": {}}, {}
    for t in TARGETS:
        y_all = patients[t].values.astype(float)
        lab = ~np.isnan(y_all)
        y = y_all[lab].astype(int)
        skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
        folds = list(skf.split(np.zeros(len(y)), y))
        oof, perm = {}, {}
        for v in variants:
            X = matrix(v, feats, meta)[lab]
            s = np.zeros(len(y))
            imp = {g: [] for g in groups_for(v, n_img)}
            for tr, te in folds:
                m = build(v, n_img).fit(X[tr], y[tr])
                s[te] = m.predict_proba(X[te])[:, 1]
                base = roc_auc_score(y[te], s[te])
                for g, cols in groups_for(v, n_img).items():      # grouped permutation importance (held-out fold)
                    drops = []
                    for r in range(3):
                        Xp = X[te].copy()
                        Xp[:, cols] = Xp[rng.permutation(len(te))][:, cols]
                        drops.append(base - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
                    imp[g].append(float(np.mean(drops)))
            oof[v] = s
            perm[v] = {g: float(np.mean(d)) for g, d in imp.items()}
        auc = {v: float(roc_auc_score(y, oof[v])) for v in variants}
        best = max(auc, key=auc.get)
        best_img = max((v for v in variants if v != "metadata"), key=auc.get)
        s = oof[best]
        ci = boot(y, s, roc_auc_score, rng)
        # SG2: does the retina add value over metadata alone? paired bootstrap of the AUROC difference
        diffs = []
        for _ in range(N_BOOT):
            i = rng.randint(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                diffs.append(roc_auc_score(y[i], oof[best_img][i]) - roc_auc_score(y[i], oof["metadata"][i]))
        added = {"variant": best_img, "delta_auroc": auc[best_img] - auc["metadata"],
                 "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
        cal = platt(s, y)
        p = cal.predict_proba(logit(s))[:, 1]
        fpr, tpr, thr = roc_curve(y, p)
        k = int(np.argmax(tpr - fpr))
        thr_p = float(thr[k])
        bins = np.clip((p * 10).astype(int), 0, 9)
        reliability = [{"bin": b / 10, "n": int((bins == b).sum()), "mean_pred": float(p[bins == b].mean()),
                        "observed": float(y[bins == b].mean())} for b in range(10) if (bins == b).sum() >= 5]
        enabled = auc[best] >= C.SYSTEMIC_ENABLE_MIN_AUROC and ci[0] >= C.SYSTEMIC_ENABLE_MIN_LOWER
        report[t] = {
            "n": int(len(y)), "n_pos": int(y.sum()), "n_unknown": int((~lab).sum()), "prevalence": float(y.mean()),
            "oof_auroc": auc, "chosen": best, "enabled": bool(enabled),
            "chosen_metrics": {"auroc": auc[best], "auroc_ci95": ci, "auprc": float(average_precision_score(y, p)),
                               "brier_calibrated": float(brier_score_loss(y, p)), "threshold": thr_p,
                               "sensitivity": float(tpr[k]), "specificity": float(1 - fpr[k])},
            "retinal_added_value": added, "permutation_importance": perm[best],
            "permutation_importance_all": perm, "reliability": reliability,
        }
        # final model on all labelled patients + occlusion baselines for per-patient explanations
        X = matrix(best, feats, meta)[lab]
        final = build(best, n_img).fit(X, y)
        bundle["heads"][t] = {"variant": best, "model": final, "platt": cal, "threshold": thr_p, "enabled": bool(enabled),
                              "backbone": best.split(":", 1)[1] if ":" in best else None,
                              "baseline_image": X[:, :n_img].mean(0) if best != "metadata" else None,
                              "baseline_meta": np.nanmedian(meta[lab], 0),
                              "oof_auroc": auc[best], "auroc_ci95": ci, "retinal_added_value": added,
                              "permutation_importance": perm[best], "n_pos": int(y.sum()), "n": int(len(y))}
        oof_rows[t] = pd.Series(np.where(lab, 0, np.nan), index=patients["patient_id"])
        oof_rows[t][lab] = p
        print(f"{t:28s} n={len(y):4d} pos={int(y.sum()):4d} | " + " ".join(f"{v.replace('RETFound_dinov2_meh_', 'RF').replace('dinov2-large_', 'DV')}={auc[v]:.3f}" for v in variants)
              + f" | chosen {best} CI {ci[0]:.3f}-{ci[1]:.3f} | retina adds {added['delta_auroc']:+.3f} "
              f"[{added['ci95'][0]:+.3f},{added['ci95'][1]:+.3f}] | enabled={enabled}", flush=True)

    joblib.dump(bundle, os.path.join(C.CKPT_DIR, "systemic_cv.joblib"))
    json.dump(report, open(os.path.join(C.METRICS_DIR, "systemic_cv.json"), "w"), indent=1)
    pd.DataFrame(oof_rows).to_csv(os.path.join(C.METRICS_DIR, "systemic_cv_oof.csv"))


if __name__ == "__main__":
    main()
