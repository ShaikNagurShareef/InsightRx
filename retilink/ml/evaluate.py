"""
Seed-ensemble, calibrate on validation, freeze thresholds, evaluate on the untouched test split.

  python -m retilink.ml.evaluate

Cascade used by the app (same rule evaluated here):
  image assessable     <=> calibrated P(quality_poor) < t_quality
  eye DR score         =  max P(dr_referable) over assessable images of that eye
  patient DR score     =  max over eyes; patient 'incomplete' if an eye has no assessable image
  patient positive     <=> patient DR score >= t_dr     (threshold chosen on val for sensitivity >= 90%)
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (average_precision_score, brier_score_loss, cohen_kappa_score, confusion_matrix,
                             roc_auc_score)

from . import config as C
from .data import load_tables
from .model import ordinal_probs_np, split_logits

BINARY_EVAL = ['quality_poor', 'dr_referable', 'edema']


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def seeds_available():
    return sorted({int(re.search(r'seed(\d+)_val_logits', p).group(1))
                   for p in glob.glob(os.path.join(C.FEAT_DIR, 'seed*_val_logits.npy'))})


def load_split_outputs(name, seeds):
    idx = pd.read_csv(os.path.join(C.FEAT_DIR, f'{name}_index.csv'))
    logits = np.mean([np.load(os.path.join(C.FEAT_DIR, f'seed{s}_{name}_logits.npy')) for s in seeds], 0)
    # per-seed embedding spaces differ -> concatenate, never average
    emb = np.concatenate([np.load(os.path.join(C.FEAT_DIR, f'seed{s}_{name}_emb.npy')).astype(np.float32)
                          for s in seeds], 1)
    return idx, split_logits(logits), emb


def fit_temperature(z, y):
    ok = ~np.isnan(y)
    z, y = z[ok], y[ok]
    nll = lambda t: -np.mean(y * np.log(sigmoid(z / t) + 1e-7) + (1 - y) * np.log(1 - sigmoid(z / t) + 1e-7))
    return float(minimize_scalar(nll, bounds=(0.25, 10), method='bounded').x)


def threshold_for_recall(score, y, target):
    """Largest threshold whose recall on positives is still >= target."""
    pos = np.sort(score[y == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[min(max(k, 0), len(pos) - 1)])


def binary_metrics(y, s, thr):
    y, s = np.asarray(y).astype(int), np.asarray(s)
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'n': int(len(y)), 'n_pos': int(y.sum()), 'auroc': float(roc_auc_score(y, s)),
            'auprc': float(average_precision_score(y, s)), 'brier': float(brier_score_loss(y, s)),
            'sensitivity': tp / max(tp + fn, 1), 'specificity': tn / max(tn + fp, 1),
            'ppv': tp / max(tp + fp, 1), 'npv': tn / max(tn + fn, 1),
            'accuracy': (tp + tn) / len(y), 'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'threshold': float(thr)}


def bootstrap_ci(groups, fn, n=1000, seed=0):
    """Patient-level bootstrap: groups is a list of per-patient row arrays."""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        pick = rng.randint(0, len(groups), len(groups))
        try:
            v = fn(np.concatenate([groups[i] for i in pick]))
        except ValueError:
            continue
        if np.isfinite(v):          # resamples with a single class give NaN AUROC -> skip
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def image_frame(idx, z, temps, images):
    df = idx.copy()
    for h in BINARY_EVAL:
        df[f'p_{h}'] = sigmoid(z[h] / temps[h])
    probs = ordinal_probs_np(z['icdr'])
    df['icdr_pred'] = probs.argmax(1)
    lab = images.set_index('image_id')
    for c in ['quality_poor', 'dr_referable', 'icdr', 'edema']:
        df[f'y_{c}'] = df['image_id'].map(lab[c]).values
    return df


def patient_cascade(df, t_q):
    """Apply the app's quality gate + max-aggregation. Returns one row per patient."""
    df = df.assign(assessable=df['p_quality_poor'] < t_q)
    rows = []
    for pid, g in df.groupby('patient_id'):
        ok = g[g['assessable']]
        eyes_ok = set(ok['eye'])
        rows.append({'patient_id': pid,
                     'score': ok['p_dr_referable'].max() if len(ok) else np.nan,
                     'complete': eyes_ok >= {'OD', 'OS'},
                     'n_assessable': int(len(ok)),
                     'y': g['y_dr_referable'].max() if g['y_dr_referable'].notna().any() else np.nan})
    return pd.DataFrame(rows)


def main():
    seeds = seeds_available()
    assert seeds, f"no exported features in {C.FEAT_DIR}; run train_image first"
    images, _ = load_tables()
    va_idx, va_z, _ = load_split_outputs('val', seeds)
    te_idx, te_z, _ = load_split_outputs('test', seeds)
    lab = images.set_index('image_id')

    temps = {h: fit_temperature(va_z[h], va_idx['image_id'].map(lab[h]).values.astype(float)) for h in BINARY_EVAL}
    va, te = image_frame(va_idx, va_z, temps, images), image_frame(te_idx, te_z, temps, images)

    # ---- frozen thresholds (validation only)
    okq = va['y_quality_poor'].notna()
    t_q = threshold_for_recall(va.loc[okq, 'p_quality_poor'].values, va.loc[okq, 'y_quality_poor'].values,
                               C.TARGET_QUALITY_RECALL)
    # 'uncertain' band below the gate: flags at most 5% of usable validation images; blocks negative summaries only
    usable = va.loc[okq & (va['y_quality_poor'] == 0), 'p_quality_poor']
    t_unc = float(min(np.percentile(usable, 95), t_q)) if len(usable) else t_q * 0.5
    # v2 rule: estimate the DR threshold on gated validation *images* (~5x more positives than the 28 validation
    # positive patients, whose 90%-sensitivity point was unstable). Revised after the v1 test evaluation; the v1
    # patient-level rule and its test metrics are kept in metrics/image_metrics_v1_patient_threshold.json.
    gated = va[(va['p_quality_poor'] < t_q) & va['y_dr_referable'].notna()]
    t_dr = threshold_for_recall(gated['p_dr_referable'].values, gated['y_dr_referable'].values, C.TARGET_DR_SENSITIVITY)
    oke = va['y_edema'].notna()
    t_ed = threshold_for_recall(va.loc[oke, 'p_edema'].values, va.loc[oke, 'y_edema'].values, 0.90)
    calib = {'seeds': seeds, 'temperatures': temps,
             'thresholds': {'quality_poor': t_q, 'quality_uncertain': t_unc, 'dr_referable_patient': t_dr,
                            'edema': t_ed},
             'threshold_version': f'val-P1-{C.EXP_NO}-v2',
             'rules': {'quality': f'recall(unusable) >= {C.TARGET_QUALITY_RECALL} on val images',
                       'dr': f'image sensitivity >= {C.TARGET_DR_SENSITIVITY} on gated val images, applied to patient max '
                             f'(v2; revised after v1 test look - see image_metrics_v1_patient_threshold.json)',
                       'edema': 'image sensitivity >= 0.90 on val'}}
    json.dump(calib, open(os.path.join(C.OUTPUT_DIR, 'calibration.json'), 'w'), indent=1)

    # ---- test metrics
    res = {'seeds': seeds, 'calibration': calib}
    for h, thr in [('quality_poor', t_q), ('dr_referable', 0.5), ('edema', t_ed)]:
        ok = te[f'y_{h}'].notna()
        res[f'image_{h}'] = binary_metrics(te.loc[ok, f'y_{h}'], te.loc[ok, f'p_{h}'], thr)
    # DR image-level among images the gate accepts, at the patient threshold
    acc = te[(te['p_quality_poor'] < t_q) & te['y_dr_referable'].notna()]
    res['image_dr_referable_gated'] = binary_metrics(acc['y_dr_referable'], acc['p_dr_referable'], t_dr)
    ok = te['y_icdr'].notna()
    res['image_icdr'] = {'qwk': float(cohen_kappa_score(te.loc[ok, 'y_icdr'].astype(int), te.loc[ok, 'icdr_pred'],
                                                        weights='quadratic')),
                         'accuracy': float((te.loc[ok, 'y_icdr'] == te.loc[ok, 'icdr_pred']).mean()),
                         'confusion': confusion_matrix(te.loc[ok, 'y_icdr'].astype(int), te.loc[ok, 'icdr_pred'],
                                                       labels=[0, 1, 2, 3, 4]).tolist()}

    pat = patient_cascade(te, t_q)
    lab_pat = pat.dropna(subset=['y'])
    scored = lab_pat.dropna(subset=['score'])
    m = binary_metrics(scored['y'], scored['score'], t_dr)
    groups = [np.array([[r.y, r.score]]) for r in scored.itertuples()]
    m['auroc_ci95'] = bootstrap_ci(groups, lambda a: roc_auc_score(a[:, 0], a[:, 1]))
    m['sensitivity_ci95'] = bootstrap_ci(groups, lambda a: ((a[:, 1] >= t_dr) & (a[:, 0] == 1)).sum() / max((a[:, 0] == 1).sum(), 1))
    m['specificity_ci95'] = bootstrap_ci(groups, lambda a: ((a[:, 1] < t_dr) & (a[:, 0] == 0)).sum() / max((a[:, 0] == 0).sum(), 1))
    m['cascade'] = {'patients_labeled': int(len(lab_pat)), 'patients_scored': int(len(scored)),
                    'unable_to_assess': int(lab_pat['score'].isna().sum()),
                    'positives_unable_to_assess': int(((lab_pat['y'] == 1) & lab_pat['score'].isna()).sum()),
                    'incomplete_eye_coverage': int((~lab_pat['complete']).sum())}
    res['patient_dr_referable'] = m

    # ---- quality gate operating characteristics on test images
    okq = te['y_quality_poor'].notna()
    rej = te['p_quality_poor'] >= t_q
    res['quality_gate'] = {'unusable_recall': float((rej & (te['y_quality_poor'] == 1)).sum() / max((te['y_quality_poor'] == 1).sum(), 1)),
                           'usable_false_rejection': float((rej & (te['y_quality_poor'] == 0)).sum() / max((te['y_quality_poor'] == 0).sum(), 1)),
                           'coverage': float((~rej[okq]).mean())}

    # ---- reliability (calibration) curves for explainability views
    def reliability(y, p):
        y, p = np.asarray(y, float), np.asarray(p, float)
        b = np.clip((p * 10).astype(int), 0, 9)
        return [{'bin': k / 10, 'n': int((b == k).sum()), 'mean_pred': float(p[b == k].mean()),
                 'observed': float(y[b == k].mean())} for k in range(10) if (b == k).sum() >= 5]
    okd = te['y_dr_referable'].notna()
    res['reliability'] = {'image_dr_referable': reliability(te.loc[okd, 'y_dr_referable'], te.loc[okd, 'p_dr_referable']),
                          'patient_dr_referable': reliability(scored['y'], scored['score'])}
    json.dump(res, open(os.path.join(C.METRICS_DIR, 'image_metrics.json'), 'w'), indent=1, default=float)
    te.to_csv(os.path.join(C.METRICS_DIR, 'test_image_preds.csv'), index=False)
    pat.to_csv(os.path.join(C.METRICS_DIR, 'test_patient_preds.csv'), index=False)
    pm = res['patient_dr_referable']
    print(f"seeds {seeds} | quality AUROC {res['image_quality_poor']['auroc']:.3f} gate recall "
          f"{res['quality_gate']['unusable_recall']:.3f} | DR image AUROC {res['image_dr_referable']['auroc']:.3f} | "
          f"DR patient AUROC {pm['auroc']:.3f} sens {pm['sensitivity']:.3f} spec {pm['specificity']:.3f} | "
          f"ICDR QWK {res['image_icdr']['qwk']:.3f} | edema AUROC {res['image_edema']['auroc']:.3f}")


if __name__ == '__main__':
    main()
